"""Latency-aware model routing and hedged requests in agents/llm_helper.py.

Why this exists: on 2026-09-11 the four Gemini models in the fallback chain were
benchmarked twice, twenty minutes apart, interleaved. gemini-3.5-flash-lite measured
10.9-17.0s in the first run and 0.95s in the second. gemini-3.6-flash — the configured
primary — answered 0 of 5 calls (503/429/timeout) while three other models were fine.

Two consequences the tests below lock in:

1. A hand-written model order cannot survive that kind of drift, so ordering is derived
   from measured latency instead.
2. Strict sequential fallback makes every request pay the full timeout of a dead model
   before reaching a live one. Hedging overlaps them, so a stalled leader costs
   `hedge_delay` rather than its whole budget.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents import llm_helper


@pytest.fixture(autouse=True)
def _clean_router_state():
    """Router state is process-global; isolate every test from its neighbours."""
    llm_helper._model_latency_ewma.clear()
    llm_helper._model_failures.clear()
    llm_helper._model_failure_streak.clear()
    llm_helper._model_timeout_at.clear()
    llm_helper._model_timeout_streak.clear()
    yield
    llm_helper._model_latency_ewma.clear()
    llm_helper._model_failures.clear()
    llm_helper._model_failure_streak.clear()
    llm_helper._model_timeout_at.clear()
    llm_helper._model_timeout_streak.clear()


class _FakeClient:
    """Stands in for genai.Client, with per-model scripted delay/outcome."""

    def __init__(self, behaviour: dict[str, tuple[float, object]]):
        self.behaviour = behaviour
        self.calls: list[str] = []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, *, model, contents, config):
        self.calls.append(model)
        delay, outcome = self.behaviour[model]
        await asyncio.sleep(delay)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


# ── Ordering ────────────────────────────────────────────────────


def test_faster_measured_model_is_tried_first():
    llm_helper._record_latency("slow-model", 9.0)
    llm_helper._record_latency("fast-model", 0.5)

    ordered = llm_helper._candidate_models(["slow-model", "fast-model"])

    assert ordered[0] == "fast-model"


def test_caller_order_breaks_ties_when_nothing_is_measured():
    """With no data, the caller's stated preference is all we have — respect it."""
    assert llm_helper._candidate_models(["a", "b", "c"]) == ["a", "b", "c"]


def test_an_unmeasured_model_outranks_one_measured_slower_than_the_prior():
    """An unknown deserves a try before a model we have watched crawl."""
    llm_helper._record_latency("known-slow", 20.0)

    ordered = llm_helper._candidate_models(["known-slow", "never-tried"])

    assert ordered[0] == "never-tried"


def test_unhealthy_models_are_demoted_below_healthy_ones():
    llm_helper._mark_model_unhealthy("broken")
    llm_helper._record_latency("broken", 0.01)  # fast, but refusing requests

    ordered = llm_helper._candidate_models(["broken", "working"])

    assert ordered == ["working", "broken"]


# ── Hedging ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_stalled_leader_does_not_block_the_next_model():
    """The 2026-09-11 failure in miniature: model A hangs, model B is healthy.

    Sequentially this costs A's entire timeout before B even starts. Hedged, it
    costs the hedge window.
    """
    client = _FakeClient(
        {
            "stalled": (30.0, "never arrives"),
            "healthy": (0.05, "real answer"),
        }
    )

    with patch.object(llm_helper, "get_genai_client", return_value=client):
        started = time.monotonic()
        result = await llm_helper.generate_gemini_content(
            contents="q",
            candidate_models=["stalled", "healthy"],
            timeout=30.0,
            hedge_delay=0.2,
        )
        elapsed = time.monotonic() - started

    assert result == "real answer"
    # Hedge window + the healthy model's own time, nowhere near the 30s budget.
    assert elapsed < 1.0, f"took {elapsed:.2f}s — the stall was not overlapped"
    assert client.calls == ["stalled", "healthy"]


@pytest.mark.asyncio
async def test_a_fast_leader_is_never_hedged():
    """Hedging costs quota. It must only trigger when the leader is actually slow."""
    client = _FakeClient(
        {
            "fast": (0.01, "answer"),
            "backup": (0.01, "should not be called"),
        }
    )

    with patch.object(llm_helper, "get_genai_client", return_value=client):
        result = await llm_helper.generate_gemini_content(
            contents="q",
            candidate_models=["fast", "backup"],
            timeout=10.0,
            hedge_delay=5.0,
        )

    assert result == "answer"
    assert client.calls == ["fast"], "a healthy leader should not have been hedged"


@pytest.mark.asyncio
async def test_an_immediate_refusal_advances_without_waiting_for_the_hedge_window():
    """A 503 is a definitive answer — there is nothing to wait out."""
    client = _FakeClient(
        {
            "refusing": (0.01, RuntimeError("503 UNAVAILABLE")),
            "healthy": (0.01, "answer"),
        }
    )

    with patch.object(llm_helper, "get_genai_client", return_value=client):
        started = time.monotonic()
        result = await llm_helper.generate_gemini_content(
            contents="q",
            candidate_models=["refusing", "healthy"],
            timeout=10.0,
            hedge_delay=5.0,
        )
        elapsed = time.monotonic() - started

    assert result == "answer"
    assert elapsed < 1.0, "a refusal should not have been treated as a pending request"


@pytest.mark.asyncio
async def test_losing_hedges_are_cancelled_not_left_running():
    """A won race must not leave a second generation billing in the background."""
    client = _FakeClient(
        {
            "stalled": (30.0, "never arrives"),
            "healthy": (0.05, "real answer"),
        }
    )

    with patch.object(llm_helper, "get_genai_client", return_value=client):
        await llm_helper.generate_gemini_content(
            contents="q",
            candidate_models=["stalled", "healthy"],
            timeout=30.0,
            hedge_delay=0.2,
        )

    await asyncio.sleep(0)
    leaked = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    assert not leaked, f"{len(leaked)} hedge task(s) still running after a winner returned"


@pytest.mark.asyncio
async def test_every_model_failing_raises_rather_than_returning_empty():
    client = _FakeClient(
        {
            "a": (0.01, RuntimeError("503 UNAVAILABLE")),
            "b": (0.01, RuntimeError("429 RESOURCE_EXHAUSTED")),
        }
    )

    with patch.object(llm_helper, "get_genai_client", return_value=client):
        with pytest.raises(Exception):
            await llm_helper.generate_gemini_content(
                contents="q", candidate_models=["a", "b"], timeout=5.0, hedge_delay=0.1
            )


@pytest.mark.asyncio
async def test_success_records_latency_so_the_next_request_orders_better():
    client = _FakeClient({"m": (0.02, "answer")})

    with patch.object(llm_helper, "get_genai_client", return_value=client):
        await llm_helper.generate_gemini_content(
            contents="q", candidate_models=["m"], timeout=5.0
        )

    assert "m" in llm_helper._model_latency_ewma


@pytest.mark.asyncio
async def test_a_timeout_is_recorded_as_slow_so_the_model_is_demoted():
    """Otherwise a model that stalls forever keeps its stale fast measurement."""
    client = _FakeClient(
        {
            "stalls": (30.0, "never"),
            "healthy": (0.01, "answer"),
        }
    )
    llm_helper._record_latency("stalls", 0.01)  # stale "it was fast once" reading

    with patch.object(llm_helper, "get_genai_client", return_value=client):
        await llm_helper.generate_gemini_content(
            contents="q",
            candidate_models=["stalls", "healthy"],
            timeout=1.0,
            hedge_delay=0.2,
        )

    assert llm_helper._model_latency_ewma["stalls"] > 0.1
