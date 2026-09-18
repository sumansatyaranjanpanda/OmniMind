"""Episodic memory — cross-thread semantic recall over past conversation turns.

The three memory tiers are deliberately non-overlapping:
  - sliding window   (agents/memory/service.py) = verbatim recent turns, same thread
  - rolling summary  (agents/memory/service.py) = compressed older turns, same thread
  - episodic recall  (this module)              = semantically relevant turns from
                                                    OTHER threads (the current thread's
                                                    older history is already covered by
                                                    the rolling summary, so it's excluded
                                                    here to avoid redundant context)

Reuses the same embedding + vector-store machinery as document retrieval
(retrieval/pinecone_client.py), scoped to a per-user "memory-{user_id}" namespace so
document search and episodic recall never mix results.
"""

from __future__ import annotations

from typing import Any

import structlog

from retrieval.pinecone_client import (
    _cosine_similarity,
    _local_vector_store,
    _sanitize_metadata,
    generate_text_embeddings,
    get_async_index,
)

logger = structlog.get_logger(__name__)

EPISODIC_TOP_K = 3
EPISODIC_MIN_SCORE = 0.75  # topical relevance, not near-duplicate (cache uses 0.90 for that)


def _memory_namespace(user_id: str) -> str:
    return f"memory-{user_id}"


async def upsert_memory_turn(
    user_id: str, thread_id: str, message_id: str, role: str, content: str
) -> None:
    """Embed and store one turn for future episodic recall. Best-effort — never raises.

    Meant to be called from a FastAPI BackgroundTask after the response is already
    sent, so this never adds latency to the request that produced the turn.
    """
    if not content.strip():
        return
    try:
        embeddings = await generate_text_embeddings([content])
        if not embeddings:
            return
        vector = {
            "id": message_id,
            "values": embeddings[0],
            "metadata": _sanitize_metadata(
                {"thread_id": thread_id, "role": role, "text": content[:500]}
            ),
        }
        namespace = _memory_namespace(user_id)
        async_index = get_async_index()
        if async_index is not None:
            await async_index.upsert(vectors=[vector], namespace=namespace)
        else:
            _local_vector_store.setdefault(namespace, []).append(vector)
    except Exception as exc:
        logger.warning("Episodic memory upsert failed (non-critical)", error=str(exc))


async def retrieve_episodic_memories(
    query: str,
    user_id: str,
    exclude_thread_id: str | None = None,
    top_k: int = EPISODIC_TOP_K,
) -> list[dict[str, Any]]:
    """Semantic search over this user's past turns, excluding the current thread."""
    if not query.strip():
        return []

    try:
        embeddings = await generate_text_embeddings([query])
        if not embeddings:
            return []
        query_embedding = embeddings[0]
        namespace = _memory_namespace(user_id)

        async_index = get_async_index()
        if async_index is not None:
            results = await async_index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k * 2,
                include_metadata=True,
            )
            matches = [{**(m.metadata or {}), "score": m.score} for m in results.matches]
        else:
            vectors = _local_vector_store.get(namespace, [])
            scored = []
            for item in vectors:
                score = _cosine_similarity(query_embedding, item["values"])
                scored.append({**item["metadata"], "score": score})
            scored.sort(key=lambda x: x["score"], reverse=True)
            matches = scored[: top_k * 2]

        filtered = [
            m
            for m in matches
            if m.get("score", 0.0) >= EPISODIC_MIN_SCORE
            and (exclude_thread_id is None or m.get("thread_id") != exclude_thread_id)
        ]
        return filtered[:top_k]
    except Exception as exc:
        logger.warning("Episodic memory retrieval failed (non-critical)", error=str(exc))
        return []
