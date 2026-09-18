"""Unit tests for the Phase 5 Adaptive Multi-Source agent system."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.state import AgentState, Citation, FusedEvidence, SubQuery


# ── State Tests ─────────────────────────────────────────────────


def test_agent_state_creation():
    """AgentState can be constructed with minimal fields."""
    state: AgentState = {
        "query": "What is attention?",
        "tenant_id": "test_tenant",
        "iteration": 0,
        "max_iterations": 2,
    }
    assert state["query"] == "What is attention?"
    assert state["tenant_id"] == "test_tenant"


def test_citation_creation():
    """Citation TypedDict can be constructed."""
    citation: Citation = {
        "marker": "[^1]",
        "chunk_id": "chunk_001",
        "text_snippet": "Attention is all you need.",
        "source_type": "document",
        "page": 3,
        "section": "3.2.1",
    }
    assert citation["marker"] == "[^1]"
    assert citation["source_type"] == "document"


def test_sub_query_creation():
    """SubQuery TypedDict can be constructed."""
    sub: SubQuery = {
        "query": "OpenAI 2026 pricing",
        "source": "web",
        "reasoning": "External competitor data",
    }
    assert sub["source"] == "web"
    assert sub["query"] == "OpenAI 2026 pricing"


def test_fused_evidence_creation():
    """FusedEvidence item can be constructed."""
    ev: FusedEvidence = {
        "source_id": 1,
        "marker": "[^1]",
        "source_type": "document",
        "title": "Doc ID: doc_01 | Section: 3.2.1",
        "content": "Attention mechanism formulas",
        "doc_id": "doc_01",
        "page": 4,
    }
    assert ev["marker"] == "[^1]"
    assert ev["source_type"] == "document"


# ── Graph Routing Tests ─────────────────────────────────────────


def test_route_after_cache_hit():
    """Cache hit routes to finalize_cached."""
    from agents.graph import route_after_cache

    state: AgentState = {"cache_hit": True, "cached_answer": "Cached response."}
    assert route_after_cache(state) == "finalize_cached"


def test_route_after_cache_miss():
    """Cache miss routes to context_rewriter."""
    from agents.graph import route_after_cache

    state: AgentState = {"cache_hit": False, "cached_answer": None}
    assert route_after_cache(state) == "context_rewriter"


def test_route_after_analyzer_direct():
    """Direct LLM intent routes to direct_llm."""
    from agents.graph import route_after_analyzer

    state: AgentState = {"intent": "direct_llm"}
    assert route_after_analyzer(state) == "direct_llm"


def test_route_after_analyzer_internal():
    """Internal RAG intent routes to retriever."""
    from agents.graph import route_after_analyzer

    state: AgentState = {"intent": "internal_rag"}
    assert route_after_analyzer(state) == "retriever"


def test_route_after_analyzer_web_still_consults_documents():
    """A web_search intent must not bypass the user's own documents.

    Routing web_search straight to the web engine made the pre-retrieval guess
    irreversible — a question the corpus could answer was answered from public pages
    instead, and nothing downstream could notice. It now fans out to both engines in
    parallel, so the documents always get their say.
    """
    from agents.graph import route_after_analyzer

    state: AgentState = {"intent": "web_search"}
    assert route_after_analyzer(state) == "hybrid_multi_engine"


def test_route_after_analyzer_graph_still_consults_documents():
    """Same for graph_rag — this is the exact route that produced the wrong-person answer."""
    from agents.graph import route_after_analyzer

    state: AgentState = {"intent": "graph_rag"}
    assert route_after_analyzer(state) == "hybrid_multi_engine"


def test_route_after_analyzer_hybrid():
    """Hybrid intent routes to parallel hybrid_multi_engine."""
    from agents.graph import route_after_analyzer

    state: AgentState = {"intent": "hybrid"}
    assert route_after_analyzer(state) == "hybrid_multi_engine"


def test_route_after_critic_verified():
    """Verified answer routes to output_guardrail."""
    from agents.graph import route_after_critic

    state: AgentState = {
        "verification_status": "VERIFIED",
        "iteration": 1,
        "max_iterations": 2,
    }
    assert route_after_critic(state) == "output_guardrail"


def test_route_after_critic_retry():
    """Unverified answer with retries remaining routes to query_rewriter."""
    from agents.graph import route_after_critic

    state: AgentState = {
        "verification_status": "PARTIALLY_VERIFIED",
        "iteration": 0,
        "max_iterations": 2,
    }
    assert route_after_critic(state) == "query_rewriter"


def test_route_after_critic_max_retries():
    """Unverified answer at max retries routes to output_guardrail."""
    from agents.graph import route_after_critic

    state: AgentState = {
        "verification_status": "PARTIALLY_VERIFIED",
        "iteration": 2,
        "max_iterations": 2,
    }
    assert route_after_critic(state) == "output_guardrail"


def test_route_after_rewriter_web():
    """Rewriter targeting web routes to web_search."""
    from agents.graph import route_after_rewriter

    state: AgentState = {
        "sub_queries": [{"query": "test", "source": "web", "reasoning": ""}],
    }
    assert route_after_rewriter(state) == "web_search"


def test_route_after_rewriter_hybrid():
    """Rewriter targeting both internal and web routes to hybrid."""
    from agents.graph import route_after_rewriter

    state: AgentState = {
        "sub_queries": [
            {"query": "test 1", "source": "internal", "reasoning": ""},
            {"query": "test 2", "source": "web", "reasoning": ""},
        ],
    }
    assert route_after_rewriter(state) == "hybrid_multi_engine"


# ── Node Logic Tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_analyzer_greeting_fast_path():
    """Greeting query triggers fast-path direct_llm classification without LLM call."""
    from agents.nodes.query_analyzer import query_analyzer_node

    state: AgentState = {"query": "Hello there!", "route_history": []}
    result = await query_analyzer_node(state)

    assert result["intent"] == "direct_llm"
    assert result["is_direct_response"] is True
    assert "query_analyzer" in result["route_history"]


@pytest.mark.asyncio
async def test_direct_llm_node_offline_greeting():
    """Direct LLM handles basic greeting offline instantly."""
    from agents.nodes.direct_llm import direct_llm_node

    state: AgentState = {"query": "Hi", "route_history": []}
    result = await direct_llm_node(state)

    assert "OmniMind" in result["final_answer"]
    assert result["verification_status"] == "DIRECT_RESPONSE"
    assert result["faithfulness_score"] == 1.0


@pytest.mark.asyncio
async def test_source_fusion_node():
    """Source Fusion unifies chunks and web results with 1-indexed markers."""
    from agents.nodes.source_fusion import source_fusion_node
    from retrieval.models import RetrievedChunk

    mock_chunk = RetrievedChunk(
        id="chk_1",
        text="Internal document content",
        dense_score=0.9,
        metadata={"section": "Intro", "page_number": 1, "doc_id": "doc_1"},
    )
    mock_web = {
        "title": "Web Title",
        "url": "https://example.com",
        "snippet": "Web result content",
    }

    state: AgentState = {
        "retrieved_chunks": [mock_chunk],
        "web_results": [mock_web],
        "route_history": [],
    }

    result = await source_fusion_node(state)
    fused = result["fused_evidence"]

    assert len(fused) == 2
    assert fused[0]["marker"] == "[^1]"
    assert fused[0]["source_type"] == "document"
    assert fused[0]["content"] == "Internal document content"

    assert fused[1]["marker"] == "[^2]"
    assert fused[1]["source_type"] == "web"
    assert fused[1]["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_cache_check_node_no_redis():
    """Cache check gracefully falls through to cache_hit=False when Redis is down."""
    from agents.nodes.cache_check import cache_check_node

    state: AgentState = {
        "query": "What is attention?",
        "tenant_id": "test_tenant",
        "route_history": [],
    }

    with patch("agents.nodes.cache_check._get_redis", AsyncMock(return_value=None)), \
         patch("agents.nodes.cache_check._get_query_embedding", AsyncMock(return_value=[0.1] * 256)):
        result = await cache_check_node(state)

    assert result["cache_hit"] is False
    assert result["cached_answer"] is None
    assert "cache_check" in result["route_history"]


@pytest.mark.asyncio
async def test_retriever_node_success():
    """Retriever node executes pipeline and writes chunks + trace_id to state."""
    from agents.nodes.retriever import retriever_node
    from retrieval.models import RetrievedChunk

    state: AgentState = {
        "query": "Scaled dot product attention",
        "tenant_id": "test_tenant",
        "sub_queries": [{"query": "Scaled dot product attention", "source": "internal", "reasoning": ""}],
        "iteration": 0,
        "route_history": [],
    }

    mock_chunks = [
        RetrievedChunk(
            id="c1",
            text="Attention equation",
            dense_score=0.95,
            metadata={"doc_id": "d1"},
        ),
    ]

    with patch("agents.nodes.retriever.execute_retrieval", return_value=(mock_chunks, "trace-123")):
        result = await retriever_node(state)

    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieval_trace_id"] == "trace-123"
    assert "retriever" in result["route_history"]


@pytest.mark.asyncio
async def test_web_search_node():
    """Web search node fetches results and writes to state."""
    from agents.nodes.web_search import web_search_node

    state: AgentState = {
        "query": "latest transformer architectures 2026",
        "sub_queries": [{"query": "latest transformer architectures 2026", "source": "web", "reasoning": ""}],
        "route_history": [],
    }

    mock_results = [
        {"title": "Result 1", "url": "https://example.com/1", "snippet": "Latest info"},
    ]

    with patch("agents.nodes.web_search.search_web", return_value=mock_results):
        result = await web_search_node(state)

    assert len(result["web_results"]) == 1
    assert "web_search" in result["route_history"]


@pytest.mark.asyncio
async def test_finalize_cached_node():
    """Finalize cached node sets correct status."""
    from agents.graph import finalize_cached_node

    state: AgentState = {
        "cached_answer": "Cached answer text",
        "route_history": [],
    }

    result = await finalize_cached_node(state)
    assert result["final_answer"] == "Cached answer text"
    assert result["verification_status"] == "CACHED"
    assert result["faithfulness_score"] == 1.0


# ── Search Web Tool Tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_search_web_tavily_success():
    """Tavily search is used when TAVILY_API_KEY is configured."""
    from agents.tools.search_web import search_web

    mock_tavily_res = [
        {"title": "Tavily Title", "url": "https://tavily.com/1", "snippet": "Tavily content"}
    ]

    with patch("agents.tools.search_web.settings.tavily_api_key", "tvly-test-key"), \
         patch("agents.tools.search_web._tavily_search", return_value=mock_tavily_res):
        results = await search_web("AI news")

    assert len(results) == 1
    assert results[0]["title"] == "Tavily Title"


@pytest.mark.asyncio
async def test_search_web_tavily_fallback_to_ddg():
    """Falls back to DuckDuckGo if Tavily call fails."""
    from agents.tools.search_web import search_web

    mock_ddg_res = [
        {"title": "DDG Title", "url": "https://ddg.com/1", "snippet": "DDG content"}
    ]

    with patch("agents.tools.search_web.settings.tavily_api_key", "tvly-test-key"), \
         patch("agents.tools.search_web._tavily_search", side_effect=Exception("API Error")), \
         patch("agents.tools.search_web._ddgs_search", return_value=mock_ddg_res):
        results = await search_web("AI news")

    assert len(results) == 1
    assert results[0]["title"] == "DDG Title"


@pytest.mark.asyncio
async def test_search_web_empty_query():
    """Empty query returns empty list immediately."""
    from agents.tools.search_web import search_web

    results = await search_web("   ")
    assert results == []


# ── Graph Compilation Test ──────────────────────────────────────


def test_graph_compiles():
    """The LangGraph state graph compiles without errors."""
    from agents.graph import build_graph

    graph = build_graph()
    compiled = graph.compile()
    assert compiled is not None


@pytest.mark.asyncio
async def test_gemini_37_high_traffic_fallback_to_36():
    """When gemini-3.7-flash encounters 429/high traffic error, it immediately falls back to gemini-3.6-flash."""
    from agents.llm_helper import generate_gemini_content

    call_history = []

    async def mock_generate_content(model, contents, config):
        call_history.append(model)
        if model == "gemini-3.7-flash":
            raise Exception("429 RESOURCE_EXHAUSTED: High traffic, server overloaded.")
        mock_resp = MagicMock()
        mock_resp.text = "Answer from fallback model"
        return mock_resp

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = mock_generate_content

    with patch("agents.llm_helper.get_genai_client", return_value=mock_client), \
         patch("agents.llm_helper.settings.gemini_model", "gemini-3.7-flash"):
        result = await generate_gemini_content(
            contents="Explain transformers",
            candidate_models=["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"],
        )

    assert result == "Answer from fallback model"
    assert "gemini-3.7-flash" in call_history
    assert "gemini-3.6-flash" in call_history
