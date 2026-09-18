"""Regression tests: a retrieval BACKEND FAILURE must not be reported as "not in your
documents".

Found during live fault-injection testing: simulating a Pinecone outage produced the
exact same answer as a genuinely empty search — "I could not find this in your
documents, and the following comes from the web" — which is false when the documents
were never actually checked. The chain that made this possible:

1. pinecone_client.search_chunks_batch caught every per-query exception internally and
   returned `[]`, indistinguishable from a real empty match.
2. retrieval.pipeline.RetrievalPipeline.retrieve() saw an empty dense_candidates list
   either way and returned `([], trace_id)` — no error signal survived.
3. agents.nodes.retriever.retriever_node's own failure detection only fires on a raised
   exception, which never happened because of (1).
4. web_fallback_node had no way to know retrieval had failed rather than legitimately
   found nothing, so it always used the same "not in your documents" disclosure.

Fixed by having search_chunks_batch raise when EVERY query in a batch hit a real
exception (as opposed to Pinecone responding normally with zero matches), which now
propagates naturally through retriever_node's existing exception handling.
"""

from __future__ import annotations

import pytest

from agents.nodes.synthesizer import WEB_FALLBACK_ERROR_NOTICE, WEB_FALLBACK_NOTICE


def test_error_and_empty_notices_are_worded_differently() -> None:
    """The two disclosures must not be interchangeable — that's the whole bug."""
    assert WEB_FALLBACK_NOTICE != WEB_FALLBACK_ERROR_NOTICE
    assert "temporarily unavailable" in WEB_FALLBACK_ERROR_NOTICE.lower()
    assert "never say" in WEB_FALLBACK_ERROR_NOTICE.lower()  # explicitly forbids the false claim
    assert "nothing relevant" in WEB_FALLBACK_NOTICE.lower()


@pytest.mark.asyncio
async def test_pinecone_total_failure_raises_instead_of_returning_empty(monkeypatch) -> None:
    """search_chunks_batch must surface a total backend failure as an exception, not `[]`."""
    import retrieval.pinecone_client as pc_mod

    class _DeadIndex:
        async def query(self, *a, **k):
            raise ConnectionError("simulated outage")

    monkeypatch.setattr(pc_mod, "get_async_index", lambda: _DeadIndex())
    async def _fake_embed(texts):
        return [[0.1] * 8 for _ in texts]

    monkeypatch.setattr(pc_mod, "generate_text_embeddings", _fake_embed)

    with pytest.raises(RuntimeError, match="Pinecone unavailable"):
        await pc_mod.search_chunks_batch(["q1", "q2"], tenant_id="t")


@pytest.mark.asyncio
async def test_pinecone_partial_failure_still_degrades_gracefully(monkeypatch) -> None:
    """One bad query among several must not take down the whole batch — only TOTAL
    failure is worth surfacing as an error; a partial failure should keep going with
    whatever succeeded, same as retriever_node's own sub-query handling."""
    import retrieval.pinecone_client as pc_mod

    calls = {"n": 0}

    class _FlakyIndex:
        async def query(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("simulated outage")

            class _Result:
                matches = []

            return _Result()

    monkeypatch.setattr(pc_mod, "get_async_index", lambda: _FlakyIndex())
    async def _fake_embed(texts):
        return [[0.1] * 8 for _ in texts]

    monkeypatch.setattr(pc_mod, "generate_text_embeddings", _fake_embed)

    result = await pc_mod.search_chunks_batch(["q1", "q2"], tenant_id="t")
    assert result == [[], []]


@pytest.mark.asyncio
async def test_web_fallback_node_flags_a_retrieval_error(monkeypatch) -> None:
    """route_after_retriever sends both cases to web_fallback; the flag it sets is
    what lets the synthesizer tell them apart."""
    import agents.graph as graph_mod

    async def _fake_web_search(state):
        return {**state, "web_results": []}

    monkeypatch.setattr(graph_mod, "web_search_node", _fake_web_search)

    errored_out = await graph_mod.web_fallback_node(
        {
            "query": "q",
            "raw_query": "q",
            "error": "Retrieval failed: all 3 queries errored",
            "sub_queries": [],
            "route_history": [],
        }
    )
    assert errored_out["retrieval_errored"] is True

    empty_out = await graph_mod.web_fallback_node(
        {
            "query": "q",
            "raw_query": "q",
            "error": None,
            "sub_queries": [],
            "route_history": [],
        }
    )
    assert empty_out["retrieval_errored"] is False
