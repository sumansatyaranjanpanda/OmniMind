"""Concurrent query-analysis + document retrieval (agents/graph.py).

Measured live 2026-09-11: the analyzer took 2-3s and retrieval a further 1-1.4s, run
strictly one after the other, so ~3.5s elapsed before synthesis could start. Retrieval
never needed the analyzer's output — and by this repo's own routing rule every intent
except small talk consults the documents regardless — so the search can start
immediately and the classification can land while it runs.

The risk this buys is reusing a search for a question it wasn't run against. The critic
retry loop and the query rewriter both re-enter the retriever with different queries,
so the guard is a value comparison of the actual query list, never a boolean "already
retrieved" flag. Most of the tests below exist to hold that line.
"""

from __future__ import annotations

import asyncio

import pytest

from agents import graph as graph_mod
from agents.nodes.retriever import retriever_node


class _Chunk:
    """Minimal stand-in for RetrievedChunk — the graph reads .rerank_score off these."""

    def __init__(self, text: str, rerank_score: float | None = 0.85):
        self.text = text
        self.rerank_score = rerank_score

    def __eq__(self, other):
        return isinstance(other, _Chunk) and self.text == other.text

    def __repr__(self):
        return f"_Chunk({self.text!r}, {self.rerank_score})"


class _Recorder:
    def __init__(self):
        self.searched: list[list[str]] = []


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()

    async def _fake_retriever(state):
        queries = [
            sq["query"] for sq in state.get("sub_queries", []) if sq.get("source") == "internal"
        ] or [state["query"]]
        rec.searched.append(queries)
        return {
            **state,
            "retrieved_chunks": [_Chunk(f"chunk-for:{q}") for q in queries],
            "retrieval_trace_id": "trace-1",
            "speculative_retrieval_queries": queries,
            "route_history": list(state.get("route_history", [])) + ["retriever"],
        }

    monkeypatch.setattr(graph_mod, "retriever_node", _fake_retriever)
    return rec


def _analyzer(intent="internal_rag", sub_queries=None, delay=0.0):
    async def _fake(state):
        if delay:
            await asyncio.sleep(delay)
        return {
            **state,
            "intent": intent,
            "complexity": "simple",
            "sub_queries": sub_queries if sub_queries is not None else [],
            "route_history": list(state.get("route_history", [])) + ["query_analyzer"],
        }

    return _fake


def _state(query="what does the SLA cover", **overrides):
    base = {"query": query, "tenant_id": "t1", "route_history": [], "iteration": 0}
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_retrieval_starts_without_waiting_for_the_analyzer(monkeypatch, recorder):
    """The whole point: a slow analyzer must not delay the search."""
    monkeypatch.setattr(graph_mod, "query_analyzer_node", _analyzer(delay=0.3))

    started = asyncio.get_event_loop().time()
    result = await graph_mod.analyze_and_retrieve_node(_state())
    elapsed = asyncio.get_event_loop().time() - started

    assert result["retrieved_chunks"]
    # Overlapped, so total is ~the analyzer's own time, not analyzer + retrieval.
    assert elapsed < 0.6
    assert recorder.searched == [["what does the SLA cover"]]


@pytest.mark.asyncio
async def test_small_talk_never_starts_a_search(monkeypatch, recorder):
    """"hi" should not cost a Pinecone query and a Cohere rerank."""
    monkeypatch.setattr(graph_mod, "query_analyzer_node", _analyzer(intent="direct_llm"))

    result = await graph_mod.analyze_and_retrieve_node(_state(query="hello"))

    assert result["intent"] == "direct_llm"
    assert recorder.searched == []


@pytest.mark.asyncio
async def test_speculative_result_is_reused_by_the_retriever_node(monkeypatch, recorder):
    """Downstream retriever must not re-run the identical search."""
    monkeypatch.setattr(graph_mod, "query_analyzer_node", _analyzer())

    after_analyze = await graph_mod.analyze_and_retrieve_node(_state())
    chunks_before = after_analyze["retrieved_chunks"]

    monkeypatch.setattr(
        "agents.nodes.retriever.execute_retrieval",
        _should_not_run,
    )
    final = await retriever_node(after_analyze)

    assert final["retrieved_chunks"] == chunks_before
    assert final["speculative_retrieval_queries"] is None


async def _should_not_run(*args, **kwargs):
    raise AssertionError("retrieval re-ran for a search that was already performed")


@pytest.mark.asyncio
async def test_a_different_query_is_not_served_from_the_speculative_result(recorder):
    """The stale-reuse hazard, stated directly.

    A retry loop re-enters the retriever with a rewritten query. Handing it the first
    query's chunks would answer the wrong question with full confidence and a passing
    faithfulness score, because the evidence really does support the text written from
    it — it just isn't evidence about what was asked.
    """
    state = _state(
        query="what is the refund window",
        speculative_retrieval_queries=["what does the SLA cover"],
        retrieved_chunks=["stale-chunk"],
    )

    ran = {"called": False}

    async def _real_retrieval(*args, **kwargs):
        ran["called"] = True
        return ([], "trace-2")

    import agents.nodes.retriever as retriever_mod

    original = retriever_mod.execute_retrieval
    retriever_mod.execute_retrieval = _real_retrieval
    try:
        await retriever_node(state)
    finally:
        retriever_mod.execute_retrieval = original

    assert ran["called"], "a changed query must trigger a fresh search"


@pytest.mark.asyncio
async def test_decomposed_sub_queries_are_not_served_by_the_raw_query_search(
    monkeypatch, recorder
):
    """If the analyzer rewrites the search, the speculative run doesn't apply."""
    monkeypatch.setattr(
        graph_mod,
        "query_analyzer_node",
        _analyzer(
            sub_queries=[{"query": "SLA uptime guarantee percentage", "source": "internal"}]
        ),
    )

    result = await graph_mod.analyze_and_retrieve_node(_state())

    # Speculation searched the raw query; the analyzer wants a different one.
    assert result["speculative_retrieval_queries"] != [
        "SLA uptime guarantee percentage"
    ]

    ran = {"called": False}

    async def _real_retrieval(*args, **kwargs):
        ran["called"] = True
        return ([], "trace-3")

    import agents.nodes.retriever as retriever_mod

    original = retriever_mod.execute_retrieval
    retriever_mod.execute_retrieval = _real_retrieval
    try:
        await retriever_node(result)
    finally:
        retriever_mod.execute_retrieval = original

    assert ran["called"], "a decomposed sub-query must be searched, not assumed covered"


@pytest.mark.asyncio
async def test_a_failed_speculation_falls_back_instead_of_breaking_the_request(monkeypatch):
    """Speculation is an optimization, never a correctness dependency."""

    async def _exploding_retriever(state):
        raise RuntimeError("pinecone unavailable")

    monkeypatch.setattr(graph_mod, "retriever_node", _exploding_retriever)
    monkeypatch.setattr(graph_mod, "query_analyzer_node", _analyzer())

    result = await graph_mod.analyze_and_retrieve_node(_state())

    assert result["intent"] == "internal_rag"
    assert not result.get("speculative_retrieval_queries")


@pytest.mark.asyncio
async def test_analyzer_failure_degrades_to_documents_only(monkeypatch, recorder):
    """A dead classifier must not take the request down with it.

    The documents were retrieved successfully; not knowing whether to *also* consult
    the web is not a reason to fail a question the corpus can already answer.
    """

    async def _exploding_analyzer(state):
        raise RuntimeError("analyzer died")

    monkeypatch.setattr(graph_mod, "query_analyzer_node", _exploding_analyzer)

    result = await graph_mod.analyze_and_retrieve_node(_state())

    assert result["intent"] == "internal_rag"
    assert result["retrieved_chunks"]


# ── Bounded wait on the analyzer (added 2026-09-11) ─────────────
#
# Measured live: retrieval finished at 2.5s with a 0.85 top score while the analyzer
# ran until 9.5s and then returned `internal_rag` with the query verbatim. Once the
# documents have answered, the analyzer is confirming a route we take anyway.


@pytest.mark.asyncio
async def test_a_slow_analyzer_does_not_hold_up_an_answered_question(monkeypatch, recorder):
    monkeypatch.setattr(graph_mod, "query_analyzer_node", _analyzer(delay=5.0))
    monkeypatch.setattr(graph_mod, "ANALYZER_GRACE_SECONDS", 0.2)

    started = asyncio.get_event_loop().time()
    result = await graph_mod.analyze_and_retrieve_node(_state())
    elapsed = asyncio.get_event_loop().time() - started

    assert elapsed < 1.5, f"waited {elapsed:.2f}s on an analyzer that had nothing to add"
    # Degrades to the documents-only route, which is this repo's prescribed default.
    assert result["intent"] == "internal_rag"
    assert result["retrieved_chunks"]


@pytest.mark.asyncio
async def test_a_slow_analyzer_IS_waited_out_when_documents_found_nothing(monkeypatch):
    """With no evidence, the graph-vs-web decision is the entire decision."""

    async def _empty_retriever(state):
        return {
            **state,
            "retrieved_chunks": [],
            "retrieval_trace_id": None,
            "speculative_retrieval_queries": [state["query"]],
            "route_history": list(state.get("route_history", [])) + ["retriever"],
        }

    monkeypatch.setattr(graph_mod, "retriever_node", _empty_retriever)
    monkeypatch.setattr(graph_mod, "query_analyzer_node", _analyzer(intent="web_search", delay=0.4))
    monkeypatch.setattr(graph_mod, "ANALYZER_GRACE_SECONDS", 0.05)

    result = await graph_mod.analyze_and_retrieve_node(_state())

    assert result["intent"] == "web_search", "routing judgement was discarded when it mattered most"


@pytest.mark.asyncio
async def test_low_scoring_chunks_do_not_count_as_having_answered(monkeypatch):
    """Evidence below the relevance floor must not short-circuit the router."""
    from agents.nodes.source_fusion import RELEVANCE_FLOOR

    class _Chunk:
        def __init__(self, score):
            self.rerank_score = score

    async def _weak_retriever(state):
        return {
            **state,
            "retrieved_chunks": [_Chunk(RELEVANCE_FLOOR - 0.01)],
            "retrieval_trace_id": "t",
            "speculative_retrieval_queries": [state["query"]],
            "route_history": list(state.get("route_history", [])) + ["retriever"],
        }

    monkeypatch.setattr(graph_mod, "retriever_node", _weak_retriever)
    monkeypatch.setattr(graph_mod, "query_analyzer_node", _analyzer(intent="web_search", delay=0.3))
    monkeypatch.setattr(graph_mod, "ANALYZER_GRACE_SECONDS", 0.05)

    result = await graph_mod.analyze_and_retrieve_node(_state())

    assert result["intent"] == "web_search"
