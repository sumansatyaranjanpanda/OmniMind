"""OmniMind Retrieval package.

Provides dense vector search (Pinecone with local Jina CLIP v2), sparse BM25 search,
query rewriting, Reciprocal Rank Fusion (RRF), and the multi-stage retrieval pipeline.
"""

from retrieval.hybrid_search import (
    InMemoryBM25Index,
    reciprocal_rank_fusion,
)
from retrieval.models import RetrievedChunk
from retrieval.query_rewriter import (
    QueryExpansionResult,
    rewrite_query,
)

__all__ = [
    "InMemoryBM25Index",
    "RetrievedChunk",
    "reciprocal_rank_fusion",
    "QueryExpansionResult",
    "rewrite_query",
]
