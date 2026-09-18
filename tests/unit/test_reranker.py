"""Unit tests for Cohere primary + FlashRank fallback cross-encoder reranker."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from reranking.reranker import (
    CascadeReranker,
    CohereReranker,
    FlashRankReranker,
    rerank_chunks,
)
from retrieval.hybrid_search import RetrievedChunk


@pytest.mark.asyncio
async def test_reranker_empty_input():
    """Empty chunk list returns empty list."""
    res = await rerank_chunks("query", [], top_k=5)
    assert res == []


@pytest.mark.asyncio
async def test_cohere_reranker_success():
    """Test Cohere Rerank API success path."""
    chunks = [
        RetrievedChunk(id="c1", text="Chunk A text"),
        RetrievedChunk(id="c2", text="Chunk B text"),
    ]

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.96},
            {"index": 0, "relevance_score": 0.42},
        ]
    }
    mock_resp.raise_for_status = MagicMock()

    reranker = CohereReranker(api_key="test-cohere-key")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
        results = await reranker.rerank("test query", chunks, top_k=2)

        assert len(results) == 2
        assert results[0].id == "c2"
        assert results[0].rerank_score == 0.96
        assert results[0].source_stage == "reranked_cohere"
        assert results[1].id == "c1"
        assert results[1].rerank_score == 0.42


@pytest.mark.asyncio
async def test_cascade_reranker_cohere_success():
    """Cascade should use Cohere when configured and successful."""
    chunks = [
        RetrievedChunk(id="c1", text="Chunk A text"),
        RetrievedChunk(id="c2", text="Chunk B text"),
    ]

    mock_cohere = AsyncMock()
    mock_cohere.api_key = "valid-key"
    mock_cohere.rerank.return_value = [
        RetrievedChunk(id="c2", text="Chunk B text", rerank_score=0.99, source_stage="reranked_cohere")
    ]

    cascade = CascadeReranker()
    cascade.cohere = mock_cohere

    results = await cascade.rerank("query", chunks, top_k=1)
    assert len(results) == 1
    assert results[0].id == "c2"
    assert results[0].source_stage == "reranked_cohere"
    mock_cohere.rerank.assert_awaited_once()


@pytest.mark.asyncio
async def test_cascade_reranker_fallback_to_flashrank():
    """Cascade should fall back to FlashRank if Cohere throws an error."""
    chunks = [
        RetrievedChunk(id="c1", text="Chunk A text"),
        RetrievedChunk(id="c2", text="Chunk B text"),
    ]

    mock_cohere = AsyncMock()
    mock_cohere.api_key = "valid-key"
    mock_cohere.rerank.side_effect = Exception("Cohere 429 Rate Limit Exceeded")

    mock_flashrank = AsyncMock()
    mock_flashrank.rerank.return_value = [
        RetrievedChunk(id="c1", text="Chunk A text", rerank_score=0.88, source_stage="reranked_flashrank")
    ]

    cascade = CascadeReranker()
    cascade.cohere = mock_cohere
    cascade.flashrank = mock_flashrank

    results = await cascade.rerank("query", chunks, top_k=1)
    assert len(results) == 1
    assert results[0].id == "c1"
    assert results[0].source_stage == "reranked_flashrank"
    mock_cohere.rerank.assert_awaited_once()
    mock_flashrank.rerank.assert_awaited_once()


@pytest.mark.asyncio
async def test_flashrank_rerank_scoring():
    """Test FlashRank local cross-encoder re-ordering."""
    chunks = [
        RetrievedChunk(id="c1", text="Cats are domestic animals."),
        RetrievedChunk(id="c2", text="Transformers rely on multi-head self-attention."),
    ]

    mock_ranker = MagicMock()
    mock_ranker.rerank.return_value = [
        {"id": "c2", "score": 0.95, "text": chunks[1].text},
        {"id": "c1", "score": 0.10, "text": chunks[0].text},
    ]

    with patch("reranking.reranker._get_flashrank_ranker", return_value=mock_ranker):
        reranker = FlashRankReranker()
        reranked = await reranker.rerank("transformer attention", chunks, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].id == "c2"
        assert reranked[0].rerank_score == 0.95
        assert reranked[0].source_stage == "reranked_flashrank"
