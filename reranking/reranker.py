"""Cross-encoder reranking module for OmniMind.

Implements a resilient cascading reranking architecture:
1. Primary: Cohere Rerank API (v3.5 / English v3.0) for enterprise-grade reasoning & accuracy.
2. Fallback: FlashRank local CPU ONNX cross-encoder (ms-marco-TinyBERT-L-2-v2) if Cohere is
   unconfigured, rate-limited, or unreachable.
3. Safety Net: Clean passthrough if all models fail.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import httpx
import structlog

from api.config import get_settings
from retrieval.models import RetrievedChunk

logger = structlog.get_logger(__name__)
settings = get_settings()

# Cached FlashRank singleton
_flashrank_model: Any = None


def _get_flashrank_ranker(model_name: str = "ms-marco-TinyBERT-L-2-v2") -> Any:
    """Lazy-load and cache the FlashRank ONNX cross-encoder model."""
    global _flashrank_model
    if _flashrank_model is None:
        try:
            from flashrank import Ranker

            logger.info("Initializing FlashRank local cross-encoder", model_name=model_name)
            _flashrank_model = Ranker(model_name=model_name, cache_dir=".cache/flashrank")
        except Exception as e:
            logger.warning("Failed to initialize FlashRank", error=str(e))
            _flashrank_model = False
    return _flashrank_model


def warm_flashrank_fallback() -> None:
    """Force-load the FlashRank model now, so the download happens at startup instead
    of during a live request.

    Caught live under concurrent load testing: on a fresh checkout, `.cache/flashrank`
    doesn't exist yet, so the first time Cohere ever fails (rate-limited, unconfigured,
    or genuinely down) a real user's request blocks on downloading a 3.26MB model file
    over the network — the exact moment a fast local fallback is supposed to matter most.
    This is synchronous and blocks whatever thread calls it; call it from a thread rather
    than the event loop's own thread.
    """
    _get_flashrank_ranker()


class BaseReranker(ABC):
    """Abstract base class for reranking implementations."""

    @abstractmethod
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Rerank candidate chunks according to query relevance."""
        pass


class CohereReranker(BaseReranker):
    """Cloud-based cross-encoder reranker using Cohere's state-of-the-art API."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout: float = 5.0,
    ):
        self.api_key = api_key or settings.cohere_api_key
        self.model_name = model_name or settings.cohere_rerank_model
        self.timeout = timeout

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Rerank passages using Cohere API."""
        if not chunks:
            return []
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is not configured")

        url = "https://api.cohere.com/v2/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Prepare document texts for Cohere
        documents = [c.text for c in chunks]

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(chunks)),
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            reranked_chunks: list[RetrievedChunk] = []

            for r in results:
                idx = r.get("index")
                score = float(r.get("relevance_score", 0.0))
                if idx is not None and 0 <= idx < len(chunks):
                    orig = chunks[idx]
                    reranked_chunks.append(
                        RetrievedChunk(
                            id=orig.id,
                            text=orig.text,
                            metadata=orig.metadata,
                            dense_score=orig.dense_score,
                            sparse_score=orig.sparse_score,
                            rrf_score=orig.rrf_score,
                            rerank_score=round(score, 6),
                            source_stage="reranked_cohere",
                        )
                    )

            logger.info(
                "Cohere cross-encoder reranking complete",
                model=self.model_name,
                candidate_count=len(chunks),
                reranked_count=len(reranked_chunks),
                top_score=reranked_chunks[0].rerank_score if reranked_chunks else None,
            )
            return reranked_chunks


class FlashRankReranker(BaseReranker):
    """Local, CPU-optimized ONNX cross-encoder reranker (ms-marco-TinyBERT-L-2-v2)."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Rerank chunks using FlashRank local cross-attention."""
        if not chunks:
            return []

        try:
            ranker = _get_flashrank_ranker(self.model_name)
            if not ranker:
                logger.debug("FlashRank unavailable; returning top_k un-reranked chunks")
                return chunks[:top_k]

            from flashrank import RerankRequest

            passages = [
                {
                    "id": c.id,
                    "text": c.text,
                    "meta": c.metadata,
                }
                for c in chunks
            ]

            rerank_request = RerankRequest(query=query, passages=passages)

            def _sync_rerank() -> list[dict[str, Any]]:
                return ranker.rerank(rerank_request)

            reranked_passages = await asyncio.to_thread(_sync_rerank)

            chunk_lookup = {c.id: c for c in chunks}
            reranked_chunks: list[RetrievedChunk] = []

            for p in reranked_passages[:top_k]:
                cid = p.get("id", "")
                orig_chunk = chunk_lookup.get(cid)
                if orig_chunk:
                    score = float(p.get("score", 0.0))
                    reranked_chunks.append(
                        RetrievedChunk(
                            id=orig_chunk.id,
                            text=orig_chunk.text,
                            metadata=orig_chunk.metadata,
                            dense_score=orig_chunk.dense_score,
                            sparse_score=orig_chunk.sparse_score,
                            rrf_score=orig_chunk.rrf_score,
                            rerank_score=round(score, 6),
                            source_stage="reranked_flashrank",
                        )
                    )

            logger.info(
                "FlashRank local cross-encoder reranking complete",
                candidate_count=len(chunks),
                reranked_count=len(reranked_chunks),
                top_score=reranked_chunks[0].rerank_score if reranked_chunks else None,
            )
            return reranked_chunks

        except Exception as e:
            logger.warning("FlashRank reranking failed; falling back to candidate order", error=str(e))
            return chunks[:top_k]


class CascadeReranker(BaseReranker):
    """Cascading reranker: Primary Cohere Rerank -> Fallback FlashRank (local)."""

    def __init__(
        self,
        cohere_model: str | None = None,
        flashrank_model: str = "ms-marco-TinyBERT-L-2-v2",
    ):
        self.cohere = CohereReranker(model_name=cohere_model)
        self.flashrank = FlashRankReranker(model_name=flashrank_model)

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        """Execute cascading reranking."""
        if not chunks:
            return []

        # 1. Try Primary: Cohere (if API key configured)
        if self.cohere.api_key:
            try:
                return await self.cohere.rerank(query=query, chunks=chunks, top_k=top_k)
            except Exception as e:
                logger.warning(
                    "Cohere reranker failed; falling back to local FlashRank",
                    error=str(e),
                )

        # 2. Fallback: Local FlashRank (ms-marco-TinyBERT-L-2-v2)
        try:
            return await self.flashrank.rerank(query=query, chunks=chunks, top_k=top_k)
        except Exception as e:
            logger.warning(
                "FlashRank reranker failed; falling back to candidate order",
                error=str(e),
            )
            return chunks[:top_k]


# Default singleton instance
_default_reranker = CascadeReranker()


async def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
    cohere_model: str | None = None,
    flashrank_model: str = "ms-marco-TinyBERT-L-2-v2",
) -> list[RetrievedChunk]:
    """Convenience function for cascading reranking."""
    if cohere_model or flashrank_model != "ms-marco-TinyBERT-L-2-v2":
        reranker = CascadeReranker(cohere_model=cohere_model, flashrank_model=flashrank_model)
        return await reranker.rerank(query=query, chunks=chunks, top_k=top_k)
    return await _default_reranker.rerank(query=query, chunks=chunks, top_k=top_k)
