"""Hybrid search engine combining dense vector search and sparse BM25 retrieval.

Uses Reciprocal Rank Fusion (RRF) to merge candidate result sets from dense
embeddings (semantic similarity) and BM25 (exact lexical matching).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from retrieval.models import RetrievedChunk

logger = structlog.get_logger(__name__)


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercased alphanumeric tokens for BM25."""
    return [t.lower() for t in re.findall(r"\b\w+\b", text) if t]


class InMemoryBM25Index:
    """In-memory BM25 index over a set of document chunks."""

    def __init__(self, chunks: list[dict[str, Any]] | list[RetrievedChunk]):
        """Initialize BM25 index from a list of chunks."""
        self.chunk_records: list[RetrievedChunk] = []
        tokenized_corpus: list[list[str]] = []

        for item in chunks:
            if isinstance(item, RetrievedChunk):
                chunk = item
            else:
                chunk = RetrievedChunk(
                    id=item.get("id", ""),
                    text=item.get("text", "") or item.get("metadata", {}).get("text", ""),
                    metadata=item.get("metadata", {}),
                )
            self.chunk_records.append(chunk)
            tokenized_corpus.append(tokenize_text(chunk.text))

        if tokenized_corpus and any(tokenized_corpus):
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        """Search the BM25 index for the top_k matching chunks."""
        if not self.bm25 or not self.chunk_records:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)
        
        # Pair chunks with scores
        scored_pairs = list(zip(self.chunk_records, doc_scores))
        # Filter out 0-score items and sort descending
        scored_pairs = [(c, float(s)) for c, s in scored_pairs if s > 0.0]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        results: list[RetrievedChunk] = []
        for chunk, score in scored_pairs[:top_k]:
            results.append(
                RetrievedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    sparse_score=score,
                    source_stage="sparse",
                )
            )

        return results


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """Merge dense and sparse search rankings using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(doc) = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)

    Args:
        dense_results: Ordered list of results from dense vector search.
        sparse_results: Ordered list of results from BM25 sparse search.
        k: Smoothing constant (default: 60, standard in IR literature).
        dense_weight: Relative weight for dense ranking.
        sparse_weight: Relative weight for sparse ranking.
        top_k: Number of fused results to return.

    Returns:
        List of unique RetrievedChunk objects sorted by fused RRF score.
    """
    chunk_map: dict[str, RetrievedChunk] = {}
    rrf_scores: dict[str, float] = {}

    # Process dense results (1-based ranking)
    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item.id or item.text
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = RetrievedChunk(
                id=item.id,
                text=item.text,
                metadata=item.metadata,
                dense_score=item.dense_score,
                sparse_score=None,
                source_stage="dense",
            )
        else:
            chunk_map[chunk_id].dense_score = item.dense_score

        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (dense_weight / (k + rank))

    # Process sparse results (1-based ranking)
    for rank, item in enumerate(sparse_results, start=1):
        chunk_id = item.id or item.text
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = RetrievedChunk(
                id=item.id,
                text=item.text,
                metadata=item.metadata,
                dense_score=None,
                sparse_score=item.sparse_score,
                source_stage="sparse",
            )
        else:
            chunk_map[chunk_id].sparse_score = item.sparse_score
            chunk_map[chunk_id].source_stage = "hybrid"

        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (sparse_weight / (k + rank))

    # Sort merged chunks by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    fused_results: list[RetrievedChunk] = []
    for cid in sorted_ids[:top_k]:
        chunk = chunk_map[cid]
        chunk.rrf_score = round(rrf_scores[cid], 6)
        fused_results.append(chunk)

    logger.debug(
        "RRF fusion complete",
        dense_count=len(dense_results),
        sparse_count=len(sparse_results),
        fused_count=len(fused_results),
    )

    return fused_results
