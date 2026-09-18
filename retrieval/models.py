"""Data models for retrieval, hybrid search, and reranking stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """Standard normalized chunk returned from any retrieval or reranking stage."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    source_stage: str = "dense"  # "dense", "sparse", "hybrid", "reranked_cohere", "reranked_flashrank"
