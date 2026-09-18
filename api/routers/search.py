"""Router for document retrieval, hybrid search, and cross-encoder reranking."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps import get_current_user
from api.models.user import User
from retrieval.pipeline import execute_retrieval

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(..., description="The user's query.")
    top_k: int = Field(5, ge=1, le=50, description="Number of final results to return.")
    enable_query_rewrite: bool = Field(
        True, description="Enable LLM-based query expansion and sub-query generation."
    )
    enable_hybrid: bool = Field(
        True, description="Enable BM25 sparse + dense vector Reciprocal Rank Fusion (RRF)."
    )
    enable_rerank: bool = Field(
        True, description="Enable Cohere/FlashRank cross-encoder reranking stage."
    )
    hybrid_alpha: float = Field(
        0.5, ge=0.0, le=1.0, description="Weight for dense search vs sparse search (0.0 to 1.0)."
    )
    trace_id: str | None = Field(
        None, description="Optional client-provided distributed trace ID."
    )


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    source_stage: str = "dense"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    total_results: int
    trace_id: str
    results: list[SearchResultItem]


@router.post("", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> SearchResponse:
    """Retrieve relevant chunks using multi-stage hybrid search & cross-encoder reranking.

    The search is strictly isolated to the current user's namespace (tenant_id)
    and fully instrumented with Langfuse distributed tracing.
    """
    tenant_id = str(current_user.id)
    chunks, trace_id = await execute_retrieval(
        query=request.query,
        tenant_id=tenant_id,
        top_k=request.top_k,
        enable_query_rewrite=request.enable_query_rewrite,
        enable_hybrid=request.enable_hybrid,
        enable_rerank=request.enable_rerank,
        hybrid_alpha=request.hybrid_alpha,
        trace_id=request.trace_id,
    )

    items = [
        SearchResultItem(
            chunk_id=c.id,
            text=c.text,
            dense_score=c.dense_score,
            sparse_score=c.sparse_score,
            rrf_score=c.rrf_score if c.rrf_score > 0.0 else None,
            rerank_score=c.rerank_score,
            source_stage=c.source_stage,
            metadata=c.metadata,
        )
        for c in chunks
    ]

    return SearchResponse(
        query=request.query,
        total_results=len(items),
        trace_id=trace_id,
        results=items,
    )
