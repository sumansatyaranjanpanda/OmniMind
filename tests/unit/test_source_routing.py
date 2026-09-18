"""Regression tests for the documents-first routing rewrite.

All of these encode one incident. A résumé was uploaded, and "who is abinash" came back
describing four unrelated public figures, correctly cited, with a passing faithfulness
score. The user's own document was never queried.

The chain was: the analyzer classified a person-question as `graph_rag` -> the knowledge
graph had no such entity -> the critic failed the empty answer -> the query rewriter's
`source` default was "web" -> web search returned same-named strangers -> the synthesizer
summarised them faithfully -> the critic passed it, because faithfulness measures
"grounded in the retrieved evidence", never "grounded in the *right* evidence".

Every link in that chain gets a test here.
"""

from __future__ import annotations

import asyncio
import random
from uuid import uuid4

import pytest

from agents.graph import route_after_analyzer, route_after_retriever
from agents.nodes.source_fusion import RELEVANCE_FLOOR
from chunking.strategies import chunk_markdown


class _Chunk:
    """Minimal stand-in for RetrievedChunk — the gate only reads rerank_score."""

    def __init__(self, rerank_score: float | None) -> None:
        self.rerank_score = rerank_score


# ── The routing decision itself ─────────────────────────────────


@pytest.mark.parametrize("intent", ["internal_rag", "graph_rag", "web_search", "hybrid"])
def test_every_information_intent_reaches_the_documents(intent: str) -> None:
    """No information-seeking route may skip document retrieval.

    `retriever` runs the documents alone; `hybrid_multi_engine` runs them in parallel with
    the graph/web engines. Any other destination means the corpus goes unread.
    """
    assert route_after_analyzer({"intent": intent}) in ("retriever", "hybrid_multi_engine")


def test_small_talk_still_skips_retrieval() -> None:
    """The fix must not make "hi" pay for a retrieval round trip."""
    assert route_after_analyzer({"intent": "direct_llm"}) == "direct_llm"


def test_unknown_intent_defaults_to_documents() -> None:
    """An unrecognised intent should read the user's documents, not the public web."""
    assert route_after_analyzer({"intent": "something_new"}) == "retriever"


# ── The evidence gate ───────────────────────────────────────────


def test_gate_synthesises_when_documents_have_relevant_evidence() -> None:
    state = {"retrieved_chunks": [_Chunk(0.42), _Chunk(0.01)]}
    assert route_after_retriever(state) == "source_fusion"


def test_gate_falls_back_to_web_only_when_the_corpus_is_empty() -> None:
    assert route_after_retriever({"retrieved_chunks": []}) == "web_fallback"


def test_gate_treats_all_below_floor_chunks_as_no_evidence() -> None:
    """Chunks the cross-encoder scored as noise must not count as an answer."""
    below = RELEVANCE_FLOOR / 2
    assert route_after_retriever({"retrieved_chunks": [_Chunk(below)]}) == "web_fallback"


def test_gate_keeps_unreranked_chunks() -> None:
    """A missing rerank score means the reranker never ran, not that the chunk is bad.

    Judging an un-reranked chunk against the cross-encoder floor would compare a raw
    cosine score to a threshold calibrated for a different scale and discard good evidence.
    """
    assert route_after_retriever({"retrieved_chunks": [_Chunk(None)]}) == "source_fusion"


# ── The rewriter's escalation default ───────────────────────────


@pytest.mark.asyncio
async def test_rewriter_retries_against_documents_when_the_llm_fails(monkeypatch) -> None:
    """The exception path defaulted to source="web" — the step that fetched the strangers."""
    import agents.nodes.query_rewriter as qr

    def _boom(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("google.genai.Client", _boom)

    out = await qr.query_rewriter_node(
        {
            "query": "who is abinash",
            "critic_feedback": "",
            "unsupported_claims": [],
            "route_history": [],
        }
    )
    assert [sq["source"] for sq in out["sub_queries"]] == ["internal"]


# ── Chunking quality ────────────────────────────────────────────


def test_heading_only_sections_are_not_indexed() -> None:
    """`## Work Experience` with nothing under it is a label, not evidence.

    The real résumé produced exactly this: an 18-character chunk that could never answer
    anything but still consumed a top-k slot.
    """
    markdown = "## Work Experience\n\n## Radical Minds\n\nBuilt an email-to-CRM microservice.\n"
    chunks = chunk_markdown(markdown, uuid4(), source_type="pdf", document_title="R.pdf")

    assert len(chunks) == 1
    assert "Radical Minds" in chunks[0].text


def test_chunks_carry_their_document_and_section() -> None:
    """A chunk must name its source so a query about the subject can match it.

    The job-history passage never repeats the person's name, so without this the
    contact-details header outranked it for "what is <name>'s experience".
    """
    markdown = "## Radical Minds\n\nBuilt an email-to-CRM microservice.\n"
    chunks = chunk_markdown(
        markdown, uuid4(), source_type="pdf", document_title="AbinashResume.pdf"
    )

    assert chunks[0].text.startswith("[AbinashResume.pdf | Radical Minds]")
    assert "email-to-CRM" in chunks[0].text


def test_chunking_without_a_title_still_works() -> None:
    """document_title is optional — callers that omit it must not crash or gain a stray prefix."""
    chunks = chunk_markdown("## A\n\nBody text here.\n", uuid4(), source_type="md")

    assert len(chunks) == 1
    assert chunks[0].text.startswith("[A]")


# ── The disclosure that was missing ─────────────────────────────


@pytest.mark.asyncio
async def test_web_fallback_marks_itself_and_asks_the_users_question(monkeypatch) -> None:
    """Falling back to the web must be flagged, and must search what the user actually asked.

    Without the flag the synthesizer cannot tell the user the answer skipped their
    documents; without the query reset the web gets a sub-query that was written for
    internal retrieval.
    """
    import agents.graph as graph_mod

    captured: dict = {}

    async def _fake_web_search(state):
        captured.update(state)
        return {**state, "web_results": []}

    monkeypatch.setattr(graph_mod, "web_search_node", _fake_web_search)

    await graph_mod.web_fallback_node(
        {
            "query": "abinash within the organization",
            "raw_query": "who is abinash",
            "sub_queries": [{"query": "abinash org role", "source": "internal"}],
            "route_history": [],
        }
    )

    assert captured["web_is_fallback"] is True
    assert captured["sub_queries"] == [
        {"query": "who is abinash", "source": "web", "reasoning": "No document evidence found"}
    ]


# ── hybrid_multi_engine result merging ──────────────────────────


@pytest.mark.asyncio
async def test_hybrid_engine_keeps_document_chunks_alongside_web_results(monkeypatch) -> None:
    """The exact bug behind a broken revenue-vs-competitor comparison.

    retriever_node, graph_retriever_node, and web_search_node are all called with the
    same pre-node state, so each returns that whole state merged with only its own
    update — every result therefore still carries every AgentState key. Merging by
    checking key presence ("retrieved_chunks" in result) found the key in *every*
    result and kept whichever ran last, discarding the retriever's real chunks in
    favor of web_search's untouched empty copy. A hybrid query asking to compare the
    user's own (internal) revenue against a competitor's (web) then came back with
    zero internal citations despite the internal retrieval succeeding.
    """
    import agents.graph as graph_mod

    async def _fake_retriever(state):
        return {**state, "retrieved_chunks": ["real_chunk"], "retrieval_trace_id": "t1"}

    async def _fake_web(state):
        return {**state, "web_results": [{"title": "competitor", "url": "https://x"}]}

    monkeypatch.setattr(graph_mod, "retriever_node", _fake_retriever)
    monkeypatch.setattr(graph_mod, "web_search_node", _fake_web)

    out = await graph_mod.hybrid_multi_engine_node(
        {
            "sub_queries": [
                {"query": "our revenue", "source": "internal"},
                {"query": "competitor revenue", "source": "web"},
            ],
            "route_history": [],
        }
    )

    assert out["retrieved_chunks"] == ["real_chunk"]
    assert out["web_results"] == [{"title": "competitor", "url": "https://x"}]


@pytest.mark.asyncio
async def test_hybrid_engine_skips_graph_branch_cleanly_when_not_requested() -> None:
    """No graph sub-query means no graph call, and graph_paths must stay empty, not stale."""
    from agents.graph import hybrid_multi_engine_node

    out = await hybrid_multi_engine_node(
        {
            "sub_queries": [{"query": "q", "source": "internal"}],
            "route_history": [],
            "tenant_id": "t",
            "query": "q",
        }
    )
    assert out["graph_paths"] == []


def test_synthesizer_discloses_a_web_fallback() -> None:
    """The fallback notice has to actually reach the model's instructions."""
    from agents.nodes.synthesizer import WEB_FALLBACK_NOTICE

    assert "could not find" in WEB_FALLBACK_NOTICE.lower()
    assert "web" in WEB_FALLBACK_NOTICE.lower()


# ── Chaos test: the merge fix must not depend on completion order ──
#
# The original bug (see test_hybrid_engine_keeps_document_chunks_alongside_web_results
# above) was invisible under normal testing because web_search always finished last in
# a fixed list — a test using synchronous return values can't tell a fix that happens
# to work from a fix that is *actually* order-independent. This drives each branch
# through a real event-loop sleep of random duration and checks the merge hundreds of
# times, so a regression that reintroduces "whichever branch finishes last wins" would
# fail intermittently rather than every time — exactly the kind of bug that survives a
# single passing CI run and then reappears in production under different load.


@pytest.mark.asyncio
async def test_hybrid_merge_is_independent_of_completion_order() -> None:
    import agents.graph as graph_mod

    rng = random.Random(0)

    def make_branch(key: str, value):
        async def _branch(state):
            await asyncio.sleep(rng.uniform(0.0, 0.01))
            return {**state, key: value}

        return _branch

    orig = (graph_mod.retriever_node, graph_mod.graph_retriever_node, graph_mod.web_search_node)
    trials = 200
    failures = []
    try:
        for i in range(trials):
            graph_mod.retriever_node = make_branch("retrieved_chunks", [f"doc_{i}"])
            graph_mod.graph_retriever_node = make_branch("graph_paths", [f"graph_{i}"])
            graph_mod.web_search_node = make_branch("web_results", [f"web_{i}"])

            out = await graph_mod.hybrid_multi_engine_node(
                {
                    "sub_queries": [
                        {"query": "q", "source": "internal"},
                        {"query": "q", "source": "graph"},
                        {"query": "q", "source": "web"},
                    ],
                    "route_history": [],
                }
            )
            if (
                out["retrieved_chunks"] != [f"doc_{i}"]
                or out["graph_paths"] != [f"graph_{i}"]
                or out["web_results"] != [f"web_{i}"]
            ):
                failures.append((i, out))
    finally:
        graph_mod.retriever_node, graph_mod.graph_retriever_node, graph_mod.web_search_node = orig

    assert not failures, f"{len(failures)}/{trials} trials lost a branch's output: {failures[:3]}"
