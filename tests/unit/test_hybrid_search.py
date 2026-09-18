"""Unit tests for BM25 lexical search and Reciprocal Rank Fusion (RRF)."""

from retrieval.hybrid_search import (
    InMemoryBM25Index,
    RetrievedChunk,
    reciprocal_rank_fusion,
    tokenize_text,
)


def test_tokenize_text():
    """Test standard alphanumeric lowercase tokenization."""
    tokens = tokenize_text("Transformer: Attention (Q, K, V) is All You Need!")
    assert "transformer" in tokens
    assert "attention" in tokens
    assert "q" in tokens
    assert "k" in tokens
    assert "v" in tokens


def test_in_memory_bm25_index():
    """Test BM25 exact and technical keyword retrieval."""
    chunks = [
        RetrievedChunk(id="c1", text="The Transformer uses multi-head self-attention mechanisms."),
        RetrievedChunk(id="c2", text="Convolutional neural networks process images hierarchically."),
        RetrievedChunk(id="c3", text="Scaled Dot-Product Attention computes softmax(QK^T / sqrt(d_k))V."),
    ]

    index = InMemoryBM25Index(chunks)
    
    # Search for specific attention math
    results = index.search("Scaled Dot-Product softmax", top_k=2)
    assert len(results) >= 1
    assert results[0].id == "c3"
    assert results[0].sparse_score is not None
    assert results[0].source_stage == "sparse"

    # Search for CNN
    cnn_results = index.search("Convolutional networks", top_k=1)
    assert len(cnn_results) == 1
    assert cnn_results[0].id == "c2"


def test_reciprocal_rank_fusion_merge():
    """Test merging disparate dense and sparse lists via RRF."""
    # Chunk A is #1 in dense, Chunk B is #2 in dense
    dense_results = [
        RetrievedChunk(id="docA", text="Text A", dense_score=0.95),
        RetrievedChunk(id="docB", text="Text B", dense_score=0.85),
    ]

    # Chunk B is #1 in sparse, Chunk C is #2 in sparse
    sparse_results = [
        RetrievedChunk(id="docB", text="Text B", sparse_score=12.4),
        RetrievedChunk(id="docC", text="Text C", sparse_score=8.1),
    ]

    # docB appears in BOTH lists, so its combined RRF score should beat docA and docC
    fused = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
        top_k=5,
    )

    assert len(fused) == 3
    assert fused[0].id == "docB"  # Top fused item
    assert fused[0].source_stage == "hybrid"
    assert fused[0].dense_score == 0.85
    assert fused[0].sparse_score == 12.4
    assert fused[0].rrf_score > fused[1].rrf_score
