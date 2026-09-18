"""Semantic cache node — Redis-backed query similarity check.

Before hitting the retrieval pipeline or any LLM, we check if a semantically similar
query has already been answered. Uses Gemini Embedding 2 (same model as our vector
store) for cosine similarity.

Cache hit at threshold ≥ 0.90 bypasses the entire pipeline → ~50ms response.

Lookup is bounded and pipelined: a per-tenant Redis sorted set (`_index_key`) tracks
the most recent CACHE_MAX_ENTRIES_PER_TENANT cache keys by recency. A check does one
ZREVRANGE to get candidate keys, then one MGET to fetch all of them in a single round
trip — not a SCAN cursor loop plus an individual GET per key. Without the cap, a check
(which runs on *every* request, cache hit or miss) would keep doing more work the
larger the cache grows — the opposite of what a cache is for.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog

from agents.state import AgentState
from api.config import get_settings

logger = structlog.get_logger(__name__)

# Bump whenever a change alters what the pipeline would answer — routing, prompts,
# chunking, retrieval. Cached entries are keyed by question, not by the code that produced
# them, so without this a fix silently keeps serving the answers it was meant to replace:
# after the documents-first routing change, "who is abinash" still replayed the old answer
# describing four unrelated strangers, because the cache had it and never re-ran the graph.
# A new version means new keys, so old entries are simply never read again and expire on
# their own TTL.
CACHE_SCHEMA_VERSION = "v3"

# Configurable similarity threshold
CACHE_SIMILARITY_THRESHOLD = 0.90
CACHE_TTL_SECONDS = 3600  # 1 hour default TTL
CACHE_KEY_PREFIX = f"omnimind:semantic_cache:{CACHE_SCHEMA_VERSION}"
CACHE_INDEX_PREFIX = f"omnimind:semantic_cache_index:{CACHE_SCHEMA_VERSION}"
CACHE_MAX_ENTRIES_PER_TENANT = 200  # bounds both the Redis scan and the in-memory fallback

# In-memory fallback dictionary: {tenant_id: [{query, answer, citations, embedding}]}
# Bounded to CACHE_MAX_ENTRIES_PER_TENANT (oldest evicted first) so a long-running
# process without Redis doesn't leak memory here indefinitely.
_in_memory_cache: dict[str, list[dict[str, Any]]] = {}


def _index_key(tenant_id: str) -> str:
    return f"{CACHE_INDEX_PREFIX}:{tenant_id}"


async def _get_redis():
    """Get the async Redis client, or None if unavailable."""
    try:
        from api.cache import get_redis

        return await get_redis()
    except Exception:
        return None


async def _get_query_embedding(query: str) -> list[float] | None:
    """Generate embedding for cache lookup using our existing embedding model."""
    try:
        from retrieval.pinecone_client import generate_text_embeddings

        embeddings = await generate_text_embeddings([query])
        return embeddings[0] if embeddings else None
    except Exception as e:
        logger.debug("Could not generate cache embedding", error=str(e))
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Fast cosine similarity between two vectors."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _best_in_memory_match(tenant_id: str, query_embedding: list[float]) -> tuple[float, str | None]:
    tenant_entries = _in_memory_cache.get(tenant_id, [])
    best_similarity = 0.0
    best_cached_answer = None
    for item in tenant_entries:
        sim = _cosine_similarity(query_embedding, item["embedding"])
        if sim > best_similarity:
            best_similarity = sim
            best_cached_answer = item["answer"]
    return best_similarity, best_cached_answer


async def cache_check_node(state: AgentState) -> AgentState:
    """Check Redis semantic cache for a similar previously-answered query.

    If a cache hit is found (similarity ≥ 0.90), sets cache_hit=True and
    cached_answer to the stored response, allowing the graph to skip
    retrieval and synthesis entirely.
    """
    query = state["query"]
    tenant_id = state.get("tenant_id", "default")
    route_history = list(state.get("route_history", []))
    route_history.append("cache_check")

    start = time.perf_counter()

    query_embedding = await _get_query_embedding(query)
    if query_embedding is None:
        return {**state, "cache_hit": False, "cached_answer": None, "route_history": route_history}

    redis = await _get_redis()
    if redis is None:
        best_similarity, best_cached_answer = _best_in_memory_match(tenant_id, query_embedding)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        if best_similarity >= CACHE_SIMILARITY_THRESHOLD and best_cached_answer:
            logger.info(
                "Semantic cache HIT (in-memory)",
                similarity=round(best_similarity, 4),
                latency_ms=elapsed_ms,
                tenant_id=tenant_id,
            )
            return {**state, "cache_hit": True, "cached_answer": best_cached_answer, "route_history": route_history}
        return {**state, "cache_hit": False, "cached_answer": None, "route_history": route_history}

    try:
        # One ZREVRANGE for the candidate key list, one MGET for all their values —
        # two round trips total regardless of how large the tenant's cache has grown,
        # instead of a SCAN cursor loop plus an individual GET per matching key.
        candidate_keys = await redis.zrevrange(_index_key(tenant_id), 0, CACHE_MAX_ENTRIES_PER_TENANT - 1)
        best_similarity = 0.0
        best_cached_answer = None
        checked = 0

        if candidate_keys:
            values = await redis.mget(candidate_keys)
            for raw in values:
                if not raw:
                    continue
                cached_data = json.loads(raw)
                cached_embedding = cached_data.get("embedding")
                if not cached_embedding:
                    continue
                similarity = _cosine_similarity(query_embedding, cached_embedding)
                checked += 1
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cached_answer = cached_data.get("answer")

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        if best_similarity >= CACHE_SIMILARITY_THRESHOLD and best_cached_answer:
            logger.info(
                "Semantic cache HIT",
                similarity=round(best_similarity, 4),
                latency_ms=elapsed_ms,
                keys_checked=checked,
                tenant_id=tenant_id,
            )
            return {**state, "cache_hit": True, "cached_answer": best_cached_answer, "route_history": route_history}

        logger.info(
            "Semantic cache MISS",
            best_similarity=round(best_similarity, 4),
            latency_ms=elapsed_ms,
            keys_checked=checked,
            tenant_id=tenant_id,
        )

    except Exception as e:
        logger.warning("Redis semantic cache check failed, falling back to in-memory cache", error=str(e))
        best_similarity, best_cached_answer = _best_in_memory_match(tenant_id, query_embedding)
        if best_similarity >= CACHE_SIMILARITY_THRESHOLD and best_cached_answer:
            return {**state, "cache_hit": True, "cached_answer": best_cached_answer, "route_history": route_history}

    return {**state, "cache_hit": False, "cached_answer": None, "route_history": route_history}


async def cache_store(
    query: str,
    answer: str,
    tenant_id: str,
    citations: list[dict] | None = None,
) -> None:
    """Store a verified answer in the semantic cache for future reuse."""
    if not answer or "could not find sufficient information" in answer.lower():
        return

    try:
        query_embedding = await _get_query_embedding(query)
        if query_embedding is None:
            return

        cache_entry = {
            "query": query,
            "answer": answer,
            "citations": citations or [],
            "embedding": query_embedding,
        }

        # In-memory fallback, bounded to the most recent N entries.
        entries = _in_memory_cache.setdefault(tenant_id, [])
        entries.append(cache_entry)
        if len(entries) > CACHE_MAX_ENTRIES_PER_TENANT:
            del entries[: len(entries) - CACHE_MAX_ENTRIES_PER_TENANT]

        redis = await _get_redis()
        if redis is not None:
            query_hash = hashlib.sha256(f"{tenant_id}:{query}".encode()).hexdigest()[:16]
            cache_key = f"{CACHE_KEY_PREFIX}:{tenant_id}:{query_hash}"
            cache_data = json.dumps(cache_entry)

            index_key = _index_key(tenant_id)
            now = time.time()
            async with redis.pipeline(transaction=False) as pipe:
                pipe.setex(cache_key, CACHE_TTL_SECONDS, cache_data)
                pipe.zadd(index_key, {cache_key: now})
                # Keep the index bounded: drop everything except the most recent N
                # members (ZREMRANGEBYRANK 0..-(N+1) removes the lowest-scored/oldest
                # entries, leaving the top N by recency).
                pipe.zremrangebyrank(index_key, 0, -(CACHE_MAX_ENTRIES_PER_TENANT + 1))
                pipe.expire(index_key, CACHE_TTL_SECONDS)
                await pipe.execute()

            logger.debug("Stored answer in Redis semantic cache", cache_key=cache_key, tenant_id=tenant_id)
        else:
            logger.debug("Stored answer in in-memory semantic cache", tenant_id=tenant_id)

    except Exception as e:
        logger.warning("Failed to store in semantic cache", error=str(e))


async def clear_cache(tenant_id: str | None = None) -> None:
    """Clear in-memory and Redis semantic cache."""
    global _in_memory_cache
    if tenant_id:
        _in_memory_cache.pop(tenant_id, None)
    else:
        _in_memory_cache.clear()

    try:
        redis = await _get_redis()
        if redis is None:
            return

        if tenant_id:
            index_key = _index_key(tenant_id)
            keys = await redis.zrange(index_key, 0, -1)
            if keys:
                await redis.delete(*keys)
            await redis.delete(index_key)
        else:
            # No per-tenant scope given — fall back to a bounded SCAN for the
            # rare full-cache-clear admin path (not on any request's hot path).
            # Unversioned base, so an admin clear also sweeps entries left behind by
            # earlier CACHE_SCHEMA_VERSIONs rather than only the current one.
            pattern = "omnimind:semantic_cache:*"
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    await redis.delete(*keys)
                if cursor == 0:
                    break
            index_pattern = f"{CACHE_INDEX_PREFIX}:*"
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor=cursor, match=index_pattern, count=100)
                if keys:
                    await redis.delete(*keys)
                if cursor == 0:
                    break
    except Exception as e:
        logger.debug("Redis cache clear failed", error=str(e))
