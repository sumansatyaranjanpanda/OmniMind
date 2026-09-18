"""Unified multi-stage retrieval pipeline for OmniMind.

Coordinates the end-to-end retrieval lifecycle with full distributed tracing:
1. Query Rewriting & Multi-Aspect Expansion (Gemini 3.5 Flash-Lite)
2. Parallel Vector (Pinecone) & Lexical (BM25) Candidate Retrieval
3. Reciprocal Rank Fusion (RRF)
4. Cross-Encoder Reranking (Cohere Primary + FlashRank Fallback)
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from observability.tracer import TraceContext
from reranking.reranker import rerank_chunks
from retrieval.hybrid_search import InMemoryBM25Index, reciprocal_rank_fusion
from retrieval.models import RetrievedChunk
from retrieval.pinecone_client import search_chunks, search_chunks_batch
from retrieval.query_rewriter import QueryExpansionResult, rewrite_query

logger = structlog.get_logger(__name__)


class RetrievalPipeline:
    """Enterprise multi-stage retrieval pipeline with full telemetry."""

    def __init__(
        self,
        candidate_k: int = 25,
        default_top_k: int = 5,
        default_alpha: float = 0.5,
    ):
        self.candidate_k = candidate_k
        self.default_top_k = default_top_k
        self.default_alpha = default_alpha

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int | None = None,
        enable_query_rewrite: bool = True,
        enable_hybrid: bool = True,
        enable_rerank: bool = True,
        hybrid_alpha: float | None = None,
        trace_id: str | None = None,
    ) -> tuple[list[RetrievedChunk], str]:
        """Execute the full multi-stage search pipeline with tracing.

        Returns:
            Tuple of (list of RetrievedChunk, trace_id string).
        """
        final_top_k = top_k or self.default_top_k
        alpha = hybrid_alpha if hybrid_alpha is not None else self.default_alpha

        tracer = TraceContext(
            name="retrieval_pipeline",
            user_id=tenant_id,
            metadata={"query": query, "top_k": final_top_k},
            trace_id=trace_id,
        )

        logger.info(
            "Starting retrieval pipeline",
            trace_id=tracer.trace_id,
            query=query,
            tenant_id=tenant_id,
            top_k=final_top_k,
            enable_query_rewrite=enable_query_rewrite,
            enable_hybrid=enable_hybrid,
            enable_rerank=enable_rerank,
        )

        # ── Step 1: Query Rewriting & Expansion ─────────────────────
        if enable_query_rewrite:
            async with tracer.span(
                name="query_rewrite",
                input_data={"query": query},
            ):
                expansion: QueryExpansionResult = await rewrite_query(query)
                queries_to_search = expansion.all_search_queries
                keywords = expansion.keywords
        else:
            expansion = QueryExpansionResult(
                original_query=query,
                rewritten_query=query,
                sub_queries=[],
                keywords=[],
            )
            queries_to_search = [query]
            keywords = []

        # ── Step 2: Dense Vector Retrieval (Pinecone) ───────────────
        async with tracer.span(
            name="dense_vector_search",
            input_data={"queries": queries_to_search, "tenant_id": tenant_id},
        ):
            dense_results_per_query = await search_chunks_batch(
                queries=queries_to_search,
                tenant_id=tenant_id,
                top_k=self.candidate_k,
            )

            # Flatten & deduplicate dense matches, preserving highest score
            chunk_dense_map: dict[str, RetrievedChunk] = {}
            for res in dense_results_per_query:
                if isinstance(res, Exception):
                    logger.warning("Dense search query task failed", error=str(res))
                    continue
                for item in res:
                    cid = str(item.get("chunk_id") or item.get("id") or "")
                    text = str(item.get("text", ""))
                    score = float(item.get("score", 0.0))
                    metadata = {k: v for k, v in item.items() if k not in ("chunk_id", "id", "score")}

                    if cid not in chunk_dense_map:
                        chunk_dense_map[cid] = RetrievedChunk(
                            id=cid,
                            text=text,
                            metadata=metadata,
                            dense_score=score,
                            source_stage="dense",
                        )
                    else:
                        if score > (chunk_dense_map[cid].dense_score or 0.0):
                            chunk_dense_map[cid].dense_score = score

            dense_candidates = sorted(
                chunk_dense_map.values(),
                key=lambda x: x.dense_score or 0.0,
                reverse=True,
            )

        if not dense_candidates:
            logger.info("No dense matches found in vector index", trace_id=tracer.trace_id)
            return [], tracer.trace_id

        # ── Step 3: Sparse (BM25) Lexical Retrieval ─────────────────
        if enable_hybrid and len(dense_candidates) > 1:
            async with tracer.span(
                name="sparse_bm25_search",
                input_data={"query": query, "keywords": keywords},
            ):
                bm25_index = InMemoryBM25Index(dense_candidates)
                bm25_query = " ".join([query] + keywords)
                sparse_candidates = bm25_index.search(bm25_query, top_k=self.candidate_k)

            # ── Step 4: Reciprocal Rank Fusion (RRF) ────────────────
            async with tracer.span(
                name="reciprocal_rank_fusion",
                metadata={"alpha": alpha, "dense_count": len(dense_candidates), "sparse_count": len(sparse_candidates)},
            ):
                fused_candidates = reciprocal_rank_fusion(
                    dense_results=dense_candidates,
                    sparse_results=sparse_candidates,
                    k=60,
                    dense_weight=alpha,
                    sparse_weight=1.0 - alpha,
                    top_k=self.candidate_k,
                )
        else:
            fused_candidates = dense_candidates[: self.candidate_k]

        # ── Step 5: Cascading Cross-Encoder Reranking ───────────────
        if enable_rerank and len(fused_candidates) > 1:
            async with tracer.span(
                name="cross_encoder_rerank",
                input_data={"candidate_count": len(fused_candidates)},
            ):
                reranked_chunks = await rerank_chunks(
                    query=expansion.rewritten_query or query,
                    chunks=fused_candidates,
                    top_k=final_top_k,
                )
                return reranked_chunks, tracer.trace_id

        return fused_candidates[:final_top_k], tracer.trace_id


# Singleton default pipeline
_pipeline = RetrievalPipeline()


async def execute_retrieval(
    query: str,
    tenant_id: str,
    top_k: int = 5,
    enable_query_rewrite: bool = True,
    enable_hybrid: bool = True,
    enable_rerank: bool = True,
    hybrid_alpha: float = 0.5,
    trace_id: str | None = None,
) -> tuple[list[RetrievedChunk], str]:
    """Execute the multi-stage retrieval pipeline with tracing."""
    return await _pipeline.retrieve(
        query=query,
        tenant_id=tenant_id,
        top_k=top_k,
        enable_query_rewrite=enable_query_rewrite,
        enable_hybrid=enable_hybrid,
        enable_rerank=enable_rerank,
        hybrid_alpha=hybrid_alpha,
        trace_id=trace_id,
    )
