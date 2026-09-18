"""Retriever node — wraps our Phase 3/4 multi-stage retrieval pipeline.

This node calls the existing RetrievalPipeline (query rewrite → dense → BM25 → RRF → cascading rerank)
for all sub-queries assigned to internal document retrieval, merging and deduplicating chunks.
"""

from __future__ import annotations

import asyncio

import structlog

from agents.state import AgentState
from retrieval.models import RetrievedChunk
from retrieval.pipeline import execute_retrieval

logger = structlog.get_logger(__name__)


async def retriever_node(state: AgentState) -> AgentState:
    """Execute multi-stage hybrid retrieval for all internal sub-queries."""
    tenant_id = state.get("tenant_id", "default")
    iteration = state.get("iteration", 0)

    # Extract internal sub-queries if planned by Query Analyzer
    sub_queries = state.get("sub_queries", [])
    internal_queries = [sq["query"] for sq in sub_queries if sq.get("source") == "internal"]

    # Fallback to main query if no internal sub-queries were explicitly tagged
    if not internal_queries:
        internal_queries = [state["query"]]

    # This exact search already ran, concurrently with the query analyzer — see
    # agents/graph.py:analyze_and_retrieve_node. Reuse it rather than paying for the
    # same hybrid search (embedding + Pinecone + BM25 + Cohere rerank) twice.
    #
    # Matched on the query list itself, not on a boolean: if the analyzer decomposed
    # the question into sub-queries that differ from the raw query it was searched
    # under, this correctly falls through and searches properly. Same for the critic's
    # retry loop, which re-enters with rewritten queries. The speculative work is only
    # ever reused for the search it actually performed.
    if state.get("speculative_retrieval_queries") == internal_queries:
        logger.info("Reusing speculative retrieval result", queries=internal_queries)
        return {**state, "speculative_retrieval_queries": None}

    route_history = list(state.get("route_history", []))
    route_history.append("retriever")

    logger.info(
        "Retriever node executing",
        internal_queries=internal_queries,
        tenant_id=tenant_id,
        iteration=iteration,
    )

    all_chunks: list[RetrievedChunk] = []
    last_trace_id: str | None = None
    seen_chunk_ids: set[str] = set()

    # Independent sub-queries hit separate Pinecone/BM25/rerank pipelines with no shared
    # state, so running them concurrently rather than one-at-a-time turns N sequential
    # round-trips into one wall-clock round-trip. return_exceptions=True so that one
    # sub-query failing (e.g. a transient Pinecone hiccup) doesn't discard results the
    # other sub-queries already retrieved successfully.
    results = await asyncio.gather(
        *[
            execute_retrieval(
                query=q,
                tenant_id=tenant_id,
                top_k=6,
                # Off: costs a full LLM round-trip (with its own multi-model fallback
                # retry chain) per sub-query, and by the time we're here the query has
                # already been coreference-resolved (context_rewriter) and decomposed
                # (query_analyzer) — voice's search_documents made this same call for
                # the identical reason (agents/voice/tools.py). Measured live 2026-09-11:
                # this single step cost 32s of a 68s /chat request on its own, mostly
                # spent retrying flaky models to produce a rewritten query that was
                # barely different from the original.
                enable_query_rewrite=False,
                enable_hybrid=True,
                enable_rerank=True,
                hybrid_alpha=0.5,
            )
            for q in internal_queries
        ],
        return_exceptions=True,
    )

    failures = 0
    for result in results:
        if isinstance(result, BaseException):
            failures += 1
            logger.warning("Sub-query retrieval failed; continuing with the rest", error=str(result))
            continue
        chunks, trace_id = result
        last_trace_id = trace_id or last_trace_id
        for chunk in chunks:
            if chunk.id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.id)
                all_chunks.append(chunk)

    if failures and not all_chunks:
        return {
            **state,
            "retrieved_chunks": [],
            "retrieval_trace_id": None,
            "error": f"Retrieval failed: all {failures} sub-quer{'y' if failures == 1 else 'ies'} errored",
            "route_history": route_history,
        }

    # Sort combined chunks by highest rerank score
    all_chunks.sort(
        key=lambda c: c.rerank_score if c.rerank_score is not None else (c.dense_score or 0.0),
        reverse=True,
    )

    logger.info(
        "Retriever node completed",
        total_chunks=len(all_chunks),
        trace_id=last_trace_id,
        failed_sub_queries=failures,
        top_score=round(all_chunks[0].rerank_score or all_chunks[0].dense_score or 0.0, 4) if all_chunks else 0.0,
    )

    return {
        **state,
        "retrieved_chunks": all_chunks,
        "retrieval_trace_id": last_trace_id,
        "route_history": route_history,
    }
