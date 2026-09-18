"""OmniMind Reranking — cascading cross-encoder rerank stage.

Phase 3: Reranking retrieved results before context assembly.
Supports Cohere Rerank (Primary) with FlashRank (ms-marco-TinyBERT-L-2-v2 Fallback).
"""

from reranking.reranker import (
    BaseReranker,
    CascadeReranker,
    CohereReranker,
    FlashRankReranker,
    rerank_chunks,
)

__all__ = [
    "BaseReranker",
    "CascadeReranker",
    "CohereReranker",
    "FlashRankReranker",
    "rerank_chunks",
]
