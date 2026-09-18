"""Unit tests for query rewriter and expansion module."""

import pytest
from unittest.mock import AsyncMock, patch

from retrieval.query_rewriter import QueryExpansionResult, rewrite_query


def test_query_expansion_result_properties():
    """Test deduplication and search query aggregation in QueryExpansionResult."""
    result = QueryExpansionResult(
        original_query="transformer attention",
        rewritten_query="how does transformer self-attention work",
        sub_queries=["scaled dot-product formula", "multi-head attention projection"],
        keywords=["transformer", "attention", "softmax"],
    )

    all_q = result.all_search_queries
    assert len(all_q) == 4
    assert "transformer attention" in all_q
    assert "how does transformer self-attention work" in all_q
    assert "scaled dot-product formula" in all_q
    assert "multi-head attention projection" in all_q


@pytest.mark.asyncio
async def test_rewrite_query_empty():
    """Empty query should return immediately with empty attributes."""
    res = await rewrite_query("   ")
    assert res.original_query == "   "
    assert res.rewritten_query == "   "
    assert res.sub_queries == []


@pytest.mark.asyncio
async def test_rewrite_query_fallback_on_exception():
    """When LLM API fails, it should gracefully fall back to heuristic word extraction."""
    with patch("agents.llm_helper.generate_gemini_content", side_effect=Exception("API Timeout")):
        res = await rewrite_query("What is the BLEU score on WMT 2014?")
        assert res.original_query == "What is the BLEU score on WMT 2014?"
        assert res.rewritten_query == "What is the BLEU score on WMT 2014?"
        assert "BLEU" in res.keywords or "score" in res.keywords
        assert len(res.all_search_queries) == 1
