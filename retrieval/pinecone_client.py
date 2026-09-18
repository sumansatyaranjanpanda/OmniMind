"""Pinecone client with cloud-native Gemini Embedding 2 (256-dim Matryoshka).

Generates text and multimodal embeddings via Google Cloud GenAI SDK (models/gemini-embedding-2)
with 256-dim Matryoshka dimensionality for ultra-fast, cost-efficient search at scale.

Every external call here uses a native async client (Gemini `client.aio.*`, Pinecone
`IndexAsyncio`) rather than `asyncio.to_thread` wrapping a synchronous SDK. Wrapping a
blocking network call in a thread still ties up one of Python's default
ThreadPoolExecutor slots (min(32, cpu_count+4)) for the call's full duration; under
concurrent request load that shared pool becomes the dominant source of tail latency
across every node in the agent graph. The async clients talk to the API directly over
the event loop, so concurrent requests don't contend for threads at all.

Supports both:
1. Cloud Managed Pinecone (when PINECONE_API_KEY and PINECONE_INDEX_HOST are set)
2. In-Memory Local Vector Store (for offline / local testing without Pinecone)
"""

import hashlib
import math
from collections import OrderedDict
from typing import Any

import structlog

from api.config import get_settings
from api.schemas.ingestion import Chunk

logger = structlog.get_logger(__name__)
settings = get_settings()

EMBEDDING_DIM = 256  # Matryoshka 256-dim dimensionality

# In-memory vector store fallback: {tenant_id: [{"id": ..., "values": [...], "metadata": {...}}]}
_local_vector_store: dict[str, list[dict[str, Any]]] = {}

# ── Pinecone Client Initialization ──────────────────────────────
pc = None
_index_host: str | None = None
_async_index = None  # lazily created — see get_async_index()

if settings.pinecone_api_key and settings.pinecone_index_host:
    try:
        raw_host = settings.pinecone_index_host.strip()
        # Clean host (strip https:// or http:// and trailing slashes for correct endpoint resolution)
        _index_host = raw_host.replace("https://", "").replace("http://", "").rstrip("/")

        from pinecone import Pinecone

        pc = Pinecone(api_key=settings.pinecone_api_key)
        logger.info("Pinecone client initialized (async)", host=_index_host)
    except Exception as e:
        logger.warning("Failed to initialize Pinecone, using in-memory vector store", error=str(e))
        pc = None
        _index_host = None
else:
    logger.debug("Pinecone credentials not configured; using in-memory local vector store")


def get_async_index():
    """Lazily create and cache the async Pinecone index client (one persistent connection pool)."""
    global _async_index
    if pc is None or _index_host is None:
        return None
    if _async_index is None:
        _async_index = pc.IndexAsyncio(host=_index_host)
    return _async_index


async def close_async_index() -> None:
    """Close the async Pinecone index client — call from app shutdown."""
    global _async_index
    if _async_index is not None:
        try:
            await _async_index.close()
        except Exception as e:
            logger.debug("Error closing async Pinecone index", error=str(e))
        _async_index = None


# Pinecone rejects any vector whose metadata exceeds 40,960 bytes. Budget slightly
# under it so the non-text fields (document_id, page, section, source_type…) always
# have room after the text has been fitted.
PINECONE_METADATA_LIMIT_BYTES = 40_960
_METADATA_BUDGET_BYTES = 38_000


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Sanitize metadata for Pinecone: correct types, and within the size limit.

    Pinecone strictly requires values to be string, number, boolean, or list of strings.
    None/null values cause a 400 Bad Request error.

    The size clamp is a backstop, not the primary defence — chunking/strategies.py
    sizes chunks so this never triggers. It exists because the failure it prevents is
    disproportionate: one oversized chunk 400s the entire batch upsert, so a single bad
    document fails ingestion for every chunk alongside it rather than just itself.
    Truncating costs the tail of one passage; raising costs the whole document.
    """
    cleaned: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, list) and all(isinstance(item, str) for item in v):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)

    text = cleaned.get("text")
    if isinstance(text, str):
        # Measured in bytes, not characters: the limit is bytes, and non-ASCII text
        # costs up to 4 bytes per character, so a character-based cap would still
        # overshoot on exactly the documents most likely to be long.
        other_bytes = sum(
            len(str(k).encode("utf-8")) + len(str(val).encode("utf-8"))
            for k, val in cleaned.items()
            if k != "text"
        )
        available = _METADATA_BUDGET_BYTES - other_bytes
        encoded = text.encode("utf-8")
        if available > 0 and len(encoded) > available:
            # Decode with errors="ignore" so a cut landing mid-codepoint drops that
            # partial character instead of raising.
            cleaned["text"] = encoded[:available].decode("utf-8", errors="ignore")
            logger.warning(
                "Chunk text truncated to fit Pinecone's metadata limit",
                original_bytes=len(encoded),
                kept_bytes=available,
                chunk_id=cleaned.get("chunk_id"),
            )

    return cleaned


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


_genai_client = None


def _get_genai_client():
    """Get or initialize the Google GenAI client."""
    global _genai_client
    if _genai_client is None and settings.gemini_api_key:
        try:
            from google import genai
            _genai_client = genai.Client(api_key=settings.gemini_api_key)
        except Exception as e:
            logger.warning("Failed to initialize Google GenAI Client", error=str(e))
    return _genai_client


def _hash_fallback_vectors(texts: list[str]) -> list[list[float]]:
    """Zero-overhead deterministic hash fallback for offline / test environments."""
    vectors = []
    for t in texts:
        seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
        vec = [(math.sin(seed + i) + 1.0) / 2.0 for i in range(EMBEDDING_DIM)]
        vectors.append(vec)
    return vectors


# Short single texts — i.e. queries — are embedded repeatedly with identical input:
# the semantic-cache check, the dense vector search, and the cache write at the end of
# the graph each embed the very same query string, at ~0.8s per call measured live
# 2026-09-11. That's ~1.6s of every request spent recomputing a value we already had.
#
# Safe to memoize because the embedding of a given string under a fixed model is
# deterministic — this cache can go stale only if the model itself changes, which is a
# process restart. Bounded and LRU so it can't grow without limit. Deliberately skips
# batches and long texts so document-ingestion traffic (thousands of distinct chunks,
# never re-embedded) can't evict the query entries this exists to serve.
_QUERY_EMBED_CACHE: OrderedDict[str, list[float]] = OrderedDict()
_QUERY_EMBED_CACHE_MAX = 512
_QUERY_EMBED_CACHE_MAX_CHARS = 512


def _query_cache_key(texts: list[str]) -> str | None:
    """Cache key for a single short text, else None (not cacheable)."""
    if len(texts) != 1:
        return None
    text = texts[0]
    if len(text) > _QUERY_EMBED_CACHE_MAX_CHARS:
        return None
    return f"{settings.gemini_embedding_model}:{settings.gemini_embedding_dim}:{text}"


async def generate_text_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for text strings using Gemini Embedding 2 (256-dim Matryoshka).

    Batches every text into a single API call rather than one call per text — both
    faster (one network round-trip instead of N) and cheaper on connection overhead.
    Single short texts (queries) are additionally memoized — see _QUERY_EMBED_CACHE.

    Each text MUST be wrapped in its own `types.Content`. Passing a bare `list[str]`
    looks like it batches but doesn't: the SDK coerces the whole list into a single
    Content with N Parts — one *document* made of N pieces — and returns exactly ONE
    embedding. Callers zip that against their inputs, so every text after the first is
    silently discarded. That bug indexed only the first chunk of every uploaded
    document and capped retrieval recall at roughly one chunk per file.
    """
    if not texts:
        return []

    cache_key = _query_cache_key(texts)
    if cache_key is not None:
        cached = _QUERY_EMBED_CACHE.get(cache_key)
        if cached is not None:
            _QUERY_EMBED_CACHE.move_to_end(cache_key)
            return [list(cached)]

    client = _get_genai_client()
    if client is not None:
        try:
            from google.genai import types

            res = await client.aio.models.embed_content(
                model=settings.gemini_embedding_model,
                contents=[types.Content(parts=[types.Part.from_text(text=t)]) for t in texts],
                config=types.EmbedContentConfig(
                    output_dimensionality=settings.gemini_embedding_dim
                ),
            )
            vectors = [e.values for e in res.embeddings]
            if len(vectors) != len(texts):
                # Never return a short list — a caller zipping it against its inputs
                # would drop data silently. Fail loudly instead.
                raise ValueError(
                    f"embedding count mismatch: sent {len(texts)} texts, got {len(vectors)} vectors"
                )
            if cache_key is not None and vectors:
                _QUERY_EMBED_CACHE[cache_key] = list(vectors[0])
                _QUERY_EMBED_CACHE.move_to_end(cache_key)
                while len(_QUERY_EMBED_CACHE) > _QUERY_EMBED_CACHE_MAX:
                    _QUERY_EMBED_CACHE.popitem(last=False)
            return vectors
        except Exception as e:
            logger.warning(
                "Gemini Embedding 2 API failed; using lightweight fallback",
                error=str(e),
                text_count=len(texts),
            )

    return _hash_fallback_vectors(texts)


async def generate_image_embeddings(image_bytes_list: list[bytes]) -> list[list[float]]:
    """Generate embeddings for images using Gemini Multimodal (256-dim Matryoshka)."""
    if not image_bytes_list:
        return []

    client = _get_genai_client()
    if client is not None:
        try:
            from google.genai import types

            # One Content per image — a bare list of Parts is read as a single
            # multi-part document and yields one embedding for the whole batch.
            # See generate_text_embeddings for the full explanation.
            res = await client.aio.models.embed_content(
                model=settings.gemini_embedding_model,
                contents=[
                    types.Content(parts=[types.Part.from_bytes(data=b, mime_type="image/png")])
                    for b in image_bytes_list
                ],
                config=types.EmbedContentConfig(
                    output_dimensionality=settings.gemini_embedding_dim
                ),
            )
            vectors = [e.values for e in res.embeddings]
            if len(vectors) != len(image_bytes_list):
                raise ValueError(
                    f"embedding count mismatch: sent {len(image_bytes_list)} images, got {len(vectors)} vectors"
                )
            return vectors
        except Exception as e:
            logger.warning(
                "Gemini Image Embedding API failed; using lightweight fallback",
                error=str(e),
                image_count=len(image_bytes_list),
            )

    # Zero-overhead deterministic fallback
    vectors = []
    for i, img_bytes in enumerate(image_bytes_list):
        seed = int(hashlib.md5(img_bytes[:100]).hexdigest()[:8], 16) if img_bytes else i
        vec = [(math.cos(seed + j) + 1.0) / 2.0 for j in range(EMBEDDING_DIM)]
        vectors.append(vec)
    return vectors


async def upsert_chunks(chunks: list[Chunk], tenant_id: str) -> None:
    """Embed and upsert text chunks to Pinecone (or in-memory store)."""
    if not chunks:
        return

    texts = [chunk.text for chunk in chunks]
    embeddings = await generate_text_embeddings(texts)

    # zip() stops at the shorter sequence, so a short embedding list would drop
    # chunks from the index without raising or logging anything. Ingestion reported
    # success while most of the document was missing — refuse to upsert a partial
    # document instead of leaving a silently incomplete index behind.
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Refusing partial upsert: {len(chunks)} chunks produced {len(embeddings)} embeddings"
        )

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        raw_meta = chunk.model_dump()
        raw_meta["document_id"] = str(raw_meta["document_id"])
        raw_meta["text"] = chunk.text
        metadata = _sanitize_metadata(raw_meta)

        vectors.append({
            "id": chunk.chunk_id,
            "values": embedding,
            "metadata": metadata,
        })

    async_index = get_async_index()
    if async_index is not None:
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            await async_index.upsert(vectors=batch, namespace=tenant_id)
        logger.info("Upserted text chunks to Pinecone", count=len(vectors), namespace=tenant_id)
    else:
        if tenant_id not in _local_vector_store:
            _local_vector_store[tenant_id] = []
        _local_vector_store[tenant_id].extend(vectors)
        logger.info("Upserted text chunks to in-memory vector store", count=len(vectors), namespace=tenant_id)


async def upsert_images(
    images: list[tuple[bytes, int | None, int]],
    document_id: str,
    source_type: str,
    tenant_id: str,
) -> None:
    """Embed and upsert raw images to Pinecone (or in-memory store)."""
    if not images:
        return

    image_bytes_list = [img_bytes for img_bytes, _, _ in images]
    embeddings = await generate_image_embeddings(image_bytes_list)

    vectors = []
    for (img_bytes, page_no, fig_idx), embedding in zip(images, embeddings):
        hash_input = f"{document_id}_img_{fig_idx}".encode("utf-8")
        chunk_id = hashlib.sha256(hash_input).hexdigest()[:16]

        raw_meta = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "text": f"[Image: Figure {fig_idx + 1} on page {page_no or 'unknown'}]",
            "section": f"Figure {fig_idx + 1}",
            "source_type": source_type,
            "content_type": "image_embedding",
        }
        if page_no is not None:
            raw_meta["page"] = page_no

        metadata = _sanitize_metadata(raw_meta)

        vectors.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": metadata,
        })

    async_index = get_async_index()
    if async_index is not None:
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            await async_index.upsert(vectors=batch, namespace=tenant_id)
        logger.info("Upserted image embeddings to Pinecone", count=len(vectors), namespace=tenant_id)
    else:
        if tenant_id not in _local_vector_store:
            _local_vector_store[tenant_id] = []
        _local_vector_store[tenant_id].extend(vectors)
        logger.info("Upserted image embeddings to in-memory vector store", count=len(vectors), namespace=tenant_id)


async def search_chunks_batch(queries: list[str], tenant_id: str, top_k: int = 5) -> list[list[dict[str, Any]]]:
    """Batch-embeds all queries in a single call and queries Pinecone concurrently.

    Raises RuntimeError when EVERY query in the batch failed with a real backend
    exception — as opposed to Pinecone responding normally with zero matches, which is
    not an error and returns `[]` per query as before. This distinction matters upstream:
    without it, a live Pinecone outage and a corpus that genuinely has nothing on the
    topic are indistinguishable by the time retriever_node sees the result, and the
    synthesizer ends up telling the user "this isn't in your documents" when the true
    story is "your documents were never actually checked". Confirmed live with a
    simulated outage before this fix existed. A PARTIAL failure (some queries errored,
    others didn't) still degrades gracefully — only total failure is worth surfacing as
    an error, matching the same all-or-nothing threshold retriever_node already applies
    to its own sub-query fan-out.
    """
    if not queries:
        return []

    query_embeddings = await generate_text_embeddings(queries)
    if not query_embeddings:
        return [[] for _ in queries]

    async_index = get_async_index()
    if async_index is not None:
        import asyncio

        async def _search_single(query_emb: list[float]) -> list[dict[str, Any]] | Exception:
            try:
                results = await async_index.query(
                    namespace=tenant_id,
                    vector=query_emb,
                    top_k=top_k,
                    include_metadata=True,
                )
                matches = []
                for match in results.matches:
                    match_data = dict(match.metadata or {})
                    match_data["score"] = match.score
                    match_data["chunk_id"] = match.id
                    matches.append(match_data)
                return matches
            except Exception as e:
                logger.warning("Pinecone query failed", error=str(e))
                return e

        raw_results = await asyncio.gather(*[_search_single(emb) for emb in query_embeddings])
        errors = [r for r in raw_results if isinstance(r, Exception)]
        if errors and len(errors) == len(raw_results):
            raise RuntimeError(f"Pinecone unavailable: all {len(errors)} quer(y/ies) failed: {errors[0]}")
        return [r if not isinstance(r, Exception) else [] for r in raw_results]

    # In-Memory Local Vector Search Fallback
    tenant_vectors = _local_vector_store.get(tenant_id, [])
    if not tenant_vectors:
        return [[] for _ in queries]

    all_results = []
    for query_embedding in query_embeddings:
        scored_items = []
        for item in tenant_vectors:
            score = _cosine_similarity(query_embedding, item["values"])
            metadata = dict(item["metadata"])
            metadata["score"] = round(score, 6)
            metadata["chunk_id"] = item["id"]
            scored_items.append(metadata)
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        all_results.append(scored_items[:top_k])

    return all_results


async def search_chunks(query: str, tenant_id: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search for the most relevant chunks in Pinecone or in-memory vector store."""
    results = await search_chunks_batch([query], tenant_id=tenant_id, top_k=top_k)
    return results[0] if results else []


async def delete_document_vectors(document_id: str, tenant_id: str) -> None:
    """Delete all vectors (text chunks + images) belonging to a document.

    Pinecone doesn't support deleting by metadata filter on all plan tiers reliably
    in serverless indexes, so this deletes by namespace + metadata filter where
    supported and falls back to a full-namespace scan for the in-memory store.
    """
    async_index = get_async_index()
    if async_index is not None:
        try:
            await async_index.delete(
                filter={"document_id": {"$eq": document_id}},
                namespace=tenant_id,
            )
            logger.info("Deleted document vectors from Pinecone", document_id=document_id, namespace=tenant_id)
        except Exception as e:
            logger.warning("Pinecone delete-by-filter failed", document_id=document_id, error=str(e))
        return

    tenant_vectors = _local_vector_store.get(tenant_id)
    if tenant_vectors:
        _local_vector_store[tenant_id] = [
            v for v in tenant_vectors if v.get("metadata", {}).get("document_id") != document_id
        ]
        logger.info("Deleted document vectors from in-memory store", document_id=document_id, tenant_id=tenant_id)
