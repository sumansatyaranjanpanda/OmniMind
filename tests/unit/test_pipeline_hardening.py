"""Regression tests for the failure modes found during the 2026-09-04 end-to-end run.

Each test here pins a specific bug that shipped silently. They are grouped by the
symptom the user actually saw, not by module, because that is what makes a future
failure recognisable.
"""

from __future__ import annotations

import asyncio

import pytest

from agents.llm_helper import _is_provider_error
from agents.nodes.critic import _uncited_assertions, critic_node
from agents.nodes.query_analyzer import is_pleasantry, looks_like_a_question, query_analyzer_node
from agents.nodes.source_fusion import RELEVANCE_FLOOR, source_fusion_node
from retrieval.models import RetrievedChunk


# ── Embeddings: the bug that indexed one chunk per document ──────────────


@pytest.mark.asyncio
async def test_upsert_refuses_partial_embedding_batch(monkeypatch):
    """A short embedding list must abort the upsert, never silently truncate.

    `zip(chunks, embeddings)` stops at the shorter sequence, so when the batched
    embed call returned one vector for six texts, five chunks vanished from the
    index while ingestion still reported success.
    """
    import retrieval.pinecone_client as pc

    chunks = [
        pc.Chunk(chunk_id=f"c{i}", document_id="11111111-1111-1111-1111-111111111111",
                 text=f"clause {i}", source_type="docx", content_type="text")
        for i in range(6)
    ]

    async def _one_short(texts):
        return [[0.1] * 256]  # the exact shape of the original bug

    monkeypatch.setattr(pc, "generate_text_embeddings", _one_short)

    with pytest.raises(ValueError, match="Refusing partial upsert"):
        await pc.upsert_chunks(chunks, tenant_id="t1")


@pytest.mark.asyncio
async def test_generate_text_embeddings_returns_one_vector_per_text(monkeypatch):
    """The embed helper must never hand back fewer vectors than it was given."""
    import retrieval.pinecone_client as pc

    class _Emb:
        def __init__(self, v):
            self.values = v

    class _Res:
        embeddings = [_Emb([0.2] * 256)]  # one vector for however many inputs

    class _Models:
        async def embed_content(self, **kwargs):
            return _Res()

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()

    monkeypatch.setattr(pc, "_get_genai_client", lambda: _Client())

    # A mismatch must fall back to the deterministic local vectors rather than
    # returning a short list that a caller would zip against its inputs.
    out = await pc.generate_text_embeddings(["a", "b", "c"])
    assert len(out) == 3


# ── Circuit breaker: don't punish a model for our own impatience ─────────


def test_local_timeout_is_not_a_provider_error():
    assert _is_provider_error(asyncio.TimeoutError()) is False
    assert _is_provider_error(TimeoutError()) is False


def test_real_provider_errors_are_recognised():
    assert _is_provider_error(Exception("429 RESOURCE_EXHAUSTED")) is True
    assert _is_provider_error(Exception("503 UNAVAILABLE. high demand")) is True
    # A status code outside any hardcoded list must still count — a real 403 was
    # being logged as a local timeout, leaving the breaker untripped against a
    # project that had been denied access entirely.
    assert _is_provider_error(Exception("403 PERMISSION_DENIED. denied access")) is True


@pytest.mark.asyncio
async def test_timeout_does_not_trip_the_circuit_breaker(monkeypatch):
    """A slow call must not sideline an otherwise healthy model.

    The breaker backs off exponentially, so marking a model unhealthy on our own
    timeout could lock a working model out for up to five minutes.
    """
    import agents.llm_helper as helper

    helper._model_failures.clear()
    helper._model_failure_streak.clear()

    class _Models:
        async def generate_content(self, **kwargs):
            await asyncio.sleep(5)

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()

    monkeypatch.setattr(helper, "get_genai_client", lambda: _Client())

    with pytest.raises(Exception):
        await helper.generate_gemini_content(
            contents="x", candidate_models=["m1"], timeout=0.01
        )

    assert "m1" not in helper._model_failures, "our own timeout must not trip the breaker"


# ── Routing: pleasantries must not run the full RAG pipeline ─────────────


@pytest.mark.parametrize(
    "text",
    [
        "hi",
        "Thanks, that's helpful. Have a good day!",  # the 70-second query
        "thanks!",
        "Good morning",
        "Hello there, hope you're well",
        "who are you",
    ],
)
def test_pleasantries_are_detected(text):
    assert is_pleasantry(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Hi, what is our PTO policy?",  # polite hat on a real question
        "Thanks — can you show me the Q3 revenue?",
        "Hello, how many days of leave do I get",
        "What is the Enterprise uptime commitment?",
        "good morning, list the support response targets",
    ],
)
def test_real_questions_are_not_mistaken_for_small_talk(text):
    assert is_pleasantry(text) is False


def test_looks_like_a_question():
    assert looks_like_a_question("What is the SLA?") is True
    assert looks_like_a_question("Tell me about pricing") is True
    assert looks_like_a_question("cheers mate") is False


@pytest.mark.asyncio
async def test_classifier_failure_falls_back_to_the_cheap_route(monkeypatch):
    """When classification breaks, small talk must not trigger retrieval.

    The old handler defaulted everything to internal_rag, so a classifier outage
    turned "thanks" into a full retrieve → synthesize → critique → web-search run.
    """
    async def _boom(**kwargs):
        raise RuntimeError("classifier down")

    monkeypatch.setattr("agents.llm_helper.generate_gemini_content", _boom)

    state = {"query": "appreciate the help", "route_history": []}
    out = await query_analyzer_node(state)  # type: ignore[arg-type]
    assert out["intent"] == "direct_llm"
    assert out["sub_queries"] == []


@pytest.mark.asyncio
async def test_classifier_failure_still_retrieves_for_real_questions(monkeypatch):
    async def _boom(**kwargs):
        raise RuntimeError("classifier down")

    monkeypatch.setattr("agents.llm_helper.generate_gemini_content", _boom)

    state = {"query": "What is the expense submission deadline", "route_history": []}
    out = await query_analyzer_node(state)  # type: ignore[arg-type]
    assert out["intent"] == "internal_rag"
    assert len(out["sub_queries"]) == 1


# ── Relevance floor: refuse rather than synthesise from noise ────────────


def _chunk(cid: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid, text=f"text {cid}", metadata={"doc_id": "d1"},
        dense_score=0.7, rerank_score=score, source_stage="reranked",
    )


@pytest.mark.asyncio
async def test_irrelevant_chunks_are_dropped_before_synthesis():
    """Chunks the cross-encoder scored as noise must not reach the synthesizer."""
    state = {
        "retrieved_chunks": [_chunk("good", 0.91), _chunk("noise", 0.02), _chunk("noise2", 0.006)],
        "graph_paths": [],
        "web_results": [],
        "route_history": [],
    }
    out = await source_fusion_node(state)  # type: ignore[arg-type]
    assert len(out["fused_evidence"]) == 1
    assert out["fused_evidence"][0]["chunk_id"] == "good"


@pytest.mark.asyncio
async def test_unreranked_chunks_are_never_filtered():
    """Raw cosine scores live on a different scale and must bypass the floor."""
    chunk = RetrievedChunk(
        id="c1", text="t", metadata={}, dense_score=0.01, rerank_score=None, source_stage="dense",
    )
    state = {"retrieved_chunks": [chunk], "graph_paths": [], "web_results": [], "route_history": []}
    out = await source_fusion_node(state)  # type: ignore[arg-type]
    assert len(out["fused_evidence"]) == 1


def test_relevance_floor_sits_between_the_measured_populations():
    """Measured: relevant chunks scored 0.59–0.98, filler scored 0.006–0.037."""
    assert 0.037 < RELEVANCE_FLOOR < 0.59


# ── Critic: stop paying for verification that has nothing to find ────────


def test_uncited_assertion_is_flagged():
    answer = "Full-time staff receive 21 days of leave. Managers approve requests in writing."
    assert len(_uncited_assertions(answer)) == 2


def test_lowercase_claim_without_numbers_still_counts():
    """A claim needs no digits or proper nouns to be a fabrication risk."""
    assert _uncited_assertions("managers must approve every request in writing") != []


def test_closing_courtesy_does_not_force_a_verification_round_trip():
    answer = "Enterprise uptime is 99.95 percent [^1]. Let me know if you need the credit table."
    assert _uncited_assertions(answer) == []


def test_fully_cited_answer_has_no_uncited_assertions():
    answer = (
        "Full-time employees receive 21 days of paid time off each year [^1]. "
        "Requests of five days or more need 21 days notice [^2]."
    )
    assert _uncited_assertions(answer) == []


def test_markdown_headings_are_not_treated_as_claims():
    """Headings must not block the fast path — the prompt asks for them."""
    answer = (
        "## Availability Commitment\n"
        "Northwind commits to 99.95 percent uptime on the Enterprise tier [^1].\n"
        "### Service Credits\n"
        "A miss below 95 percent returns a full monthly credit [^2]."
    )
    assert _uncited_assertions(answer) == []


@pytest.mark.asyncio
async def test_critic_skips_the_llm_when_every_claim_is_cited(monkeypatch):
    """The common good path must not cost a model round trip."""
    async def _fail(**kwargs):
        raise AssertionError("critic should not have called the LLM")

    monkeypatch.setattr("agents.llm_helper.generate_gemini_content", _fail)

    state = {
        "draft_answer": "Enterprise uptime is committed at 99.95 percent [^1].",
        "fused_evidence": [{"source_id": 1, "content": "99.95 percent on the Enterprise tier"}],
        "citations": [{"marker": "[^1]"}],
        "iteration": 0,
        "max_iterations": 2,
        "route_history": [],
    }
    out = await critic_node(state)  # type: ignore[arg-type]
    assert out["verification_status"] == "VERIFIED"
    assert out["faithfulness_score"] == 1.0


@pytest.mark.asyncio
async def test_critic_skips_the_llm_when_there_is_no_evidence(monkeypatch):
    """Verifying an answer against zero sources can only produce a guess."""
    async def _fail(**kwargs):
        raise AssertionError("critic should not have called the LLM")

    monkeypatch.setattr("agents.llm_helper.generate_gemini_content", _fail)

    state = {
        "draft_answer": "I could not find sufficient information to answer this question.",
        "fused_evidence": [],
        "retrieved_chunks": [],
        "web_results": [],
        "citations": [],
        "iteration": 1,
        "max_iterations": 2,
        "route_history": [],
    }
    out = await critic_node(state)  # type: ignore[arg-type]
    assert out["verification_status"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_citation_from_web_source_does_not_crash_on_missing_chunk_id(monkeypatch):
    """Web and graph evidence have no chunk_id — citing them must not 500.

    `ev.get("chunk_id", "")` only falls back when the key is absent, not when it is
    present and explicitly None. Web/graph evidence sets it to None, so every response
    that cited a web or graph source raised a pydantic ValidationError against
    Citation's non-nullable `chunk_id: str` and the request 500'd — this only surfaced
    once web search actually got exercised end-to-end.
    """
    from agents.nodes.synthesizer import synthesizer_node

    async def _answer(**kwargs):
        return "Boston Dynamics' CEO is Robert Playter [^1]."

    monkeypatch.setattr("agents.llm_helper.generate_gemini_content", _answer)

    state = {
        "query": "Who is the CEO of Boston Dynamics?",
        "fused_evidence": [
            {"source_id": 1, "marker": "[^1]", "source_type": "web",
             "title": "Boston Dynamics", "content": "Robert Playter is CEO.",
             "url": "https://example.com", "chunk_id": None, "doc_id": None,
             "page": None, "section": None}
        ],
        "route_history": [],
    }
    out = await synthesizer_node(state)  # type: ignore[arg-type]

    assert len(out["citations"]) == 1
    assert out["citations"][0]["chunk_id"] == ""  # must be a string, never None


@pytest.mark.asyncio
async def test_failed_synthesis_never_presents_a_raw_chunk_as_an_answer(monkeypatch):
    """A model outage must read as an outage, not as a confident reply.

    The fallback used to return the top evidence chunk verbatim. Because that chunk
    usually contains the very fact being asked about, the failure was invisible —
    the reply looked like a correct, if oddly formatted, answer.
    """
    from agents.nodes.synthesizer import synthesizer_node

    async def _down(**kwargs):
        raise RuntimeError("403 PERMISSION_DENIED")

    monkeypatch.setattr("agents.llm_helper.generate_gemini_content", _down)

    chunk_text = "Full-time employees accrue 21 days of paid time off annually."
    state = {
        "query": "How many PTO days?",
        "fused_evidence": [
            {"source_id": 1, "marker": "[^1]", "source_type": "document",
             "title": "Handbook", "content": chunk_text, "section": "Paid Time Off"}
        ],
        "route_history": [],
    }
    out = await synthesizer_node(state)  # type: ignore[arg-type]

    assert out["synthesis_failed"] is True
    assert out["citations"] == []
    assert out["draft_answer"] != chunk_text
    assert "could not generate an answer" in out["draft_answer"].lower()


@pytest.mark.asyncio
async def test_critic_does_not_retry_when_the_model_is_down(monkeypatch):
    """Retrying synthesis against an unreachable model just adds latency to an outage."""
    async def _fail(**kwargs):
        raise AssertionError("critic should not have called the LLM")

    monkeypatch.setattr("agents.llm_helper.generate_gemini_content", _fail)

    state = {
        "draft_answer": "I could not generate an answer just now.",
        "synthesis_failed": True,
        "fused_evidence": [{"source_id": 1, "content": "something"}],
        "citations": [],
        "iteration": 0,
        "max_iterations": 2,
        "route_history": [],
    }
    out = await critic_node(state)  # type: ignore[arg-type]
    assert out["verification_status"] == "INSUFFICIENT_EVIDENCE"
    # Iteration budget exhausted so route_after_critic finalises instead of looping.
    assert out["iteration"] >= state["max_iterations"]


@pytest.mark.asyncio
async def test_critic_still_runs_the_llm_on_a_partly_uncited_answer(monkeypatch):
    """An answer with a bare claim must still face the real verifier."""
    called = {"n": 0}

    async def _judge(**kwargs):
        called["n"] += 1
        return '{"faithfulness_score": 0.9, "verification_status": "VERIFIED", "feedback": "ok", "unsupported_claims": [], "verified_claims": []}'

    monkeypatch.setattr("agents.llm_helper.generate_gemini_content", _judge)

    state = {
        "draft_answer": "Enterprise uptime is 99.95 percent [^1]. Standard customers get 30 minute responses.",
        "fused_evidence": [{"source_id": 1, "content": "99.95 percent"}],
        "citations": [{"marker": "[^1]"}],
        "iteration": 0,
        "max_iterations": 2,
        "route_history": [],
    }
    await critic_node(state)  # type: ignore[arg-type]
    assert called["n"] == 1
