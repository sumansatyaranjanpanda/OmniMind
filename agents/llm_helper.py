"""Unified Gemini LLM helper supporting Gemini 3.7 Flash with thinking budget and resilient fallback.

Uses the native async client (`client.aio.models.*`) throughout — not `asyncio.to_thread`
wrapping the sync SDK. Wrapping a blocking network call in a thread still occupies one
of Python's default ThreadPoolExecutor slots (min(32, cpu_count+4)) for the full duration
of the call; under real concurrency every node in the agent graph competing for that same
pool becomes the dominant source of tail latency. The native async client talks to the
API directly over the event loop, so concurrent requests don't contend for threads at all.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

import structlog
from google import genai
from google.genai import types

from api.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

_client: genai.Client | None = None


def get_genai_client() -> genai.Client | None:
    """Singleton getter for Google GenAI SDK Client."""
    global _client
    if _client is None and settings.gemini_api_key:
        try:
            _client = genai.Client(api_key=settings.gemini_api_key)
        except Exception as e:
            logger.warning("Failed to initialize GenAI client", error=str(e))
    return _client


def clean_json_response(raw_text: str) -> str:
    """Clean markdown triple-backticks from JSON responses."""
    import re
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


_model_failures: dict[str, float] = {}
_model_failure_streak: dict[str, int] = {}
BASE_COOLDOWN_SECONDS = 30.0  # Recovers fast from a single transient blip
MAX_COOLDOWN_SECONDS = 300.0  # ...but backs off up to 5min under a sustained outage

# A *second*, separate breaker for our own local timeouts. Provider errors and
# timeouts are deliberately NOT the same signal (see _is_provider_error) — but
# treating every timeout as forever-forgivable was itself wrong. Measured live
# 2026-09-11: gemini-3.6-flash needed 7-10s just to say "hello" (unavoidable
# implicit thinking overhead it won't let us disable — see _build_config), which
# blew straight through synthesizer's 10s budget on nearly every call. Without
# this, a model that structurally cannot meet our timeout gets retried at full
# timeout cost on every single subsequent request, forever. One timeout is still
# forgiven outright (matches the original reasoning: don't punish a single slow
# response). Only a STREAK trips a short, flat, non-escalating cooldown — short
# because "too slow for our SLA" should be re-probed soon in case the provider
# recovers, and flat because repeatedly re-tripping it is fine and cheap.
_model_timeout_at: dict[str, float] = {}
_model_timeout_streak: dict[str, int] = {}
TIMEOUT_COOLDOWN_SECONDS = 20.0
TIMEOUT_STREAK_THRESHOLD = 2


def _is_model_healthy(model: str) -> bool:
    """Check if model is healthy or currently in cooldown.

    Cooldown backs off exponentially with consecutive failures (30s, 60s, 120s,
    240s, capped at 5min) rather than a fixed 30s. A provider-side outage (e.g.
    Gemini returning 503 "high demand" for several minutes) otherwise means every
    request pays the same full timeout retesting a model that's still down —
    a fixed 30s window is fine for a single blip but keeps re-paying that tax
    throughout a longer outage.
    """
    failed_at = _model_failures.get(model)
    if failed_at is not None:
        streak = _model_failure_streak.get(model, 1)
        cooldown = min(MAX_COOLDOWN_SECONDS, BASE_COOLDOWN_SECONDS * (2 ** (streak - 1)))
        if (time.time() - failed_at) < cooldown:
            return False

    timeout_at = _model_timeout_at.get(model)
    if timeout_at is not None and _model_timeout_streak.get(model, 0) >= TIMEOUT_STREAK_THRESHOLD:
        if (time.time() - timeout_at) < TIMEOUT_COOLDOWN_SECONDS:
            return False

    return True


def _mark_model_unhealthy(model: str) -> None:
    """Trip circuit breaker for model, extending the cooldown on repeated failures."""
    _model_failures[model] = time.time()
    _model_failure_streak[model] = _model_failure_streak.get(model, 0) + 1


def _mark_model_timed_out(model: str) -> None:
    """Record a local timeout, tripping a short cooldown only after a streak."""
    _model_timeout_at[model] = time.time()
    _model_timeout_streak[model] = _model_timeout_streak.get(model, 0) + 1


def _mark_model_healthy(model: str) -> None:
    """Reset both circuit breakers for model."""
    _model_failures.pop(model, None)
    _model_failure_streak.pop(model, None)
    _model_timeout_at.pop(model, None)
    _model_timeout_streak.pop(model, None)


# Observed end-to-end latency per model, exponentially weighted so the router tracks
# what the provider is doing *now* rather than what it did at boot.
#
# Why this exists at all: on 2026-09-11 the same four models were benchmarked twice,
# twenty minutes apart, interleaved to control for drift. gemini-3.5-flash-lite measured
# 10.9-17.0s in the first run and 0.95s in the second; gemini-3.6-flash — the configured
# *primary* — went 0-for-5 (503/429/timeout) while three other models answered fine.
# Provider latency here swings by an order of magnitude within the hour, so ANY
# hand-written model order is stale almost immediately. Two model orderings were
# hard-coded in this repo on the strength of a single benchmark and both were already
# wrong by the next measurement.
#
# The candidate list a caller passes therefore means "these models are acceptable for
# this task" (a quality judgement, which belongs with the caller). Which of them to try
# FIRST is a latency judgement, and that is made from measurements, here.
_model_latency_ewma: dict[str, float] = {}
# Asymmetric on purpose: react fast when a model gets slower, recover slowly when it
# looks faster again. A model degrading mid-shift is something every subsequent request
# pays for until the router notices, so bad news should move the estimate most of the
# way immediately. Good news is cheap to wait on, and averaging it in gently prevents
# one lucky fast response from re-promoting a model that is still mostly struggling.
EWMA_ALPHA_WORSE = 0.6
EWMA_ALPHA_BETTER = 0.2
# Neutral prior for a model we've never called: better than a model measured slower
# than this, worse than one measured faster, so an unknown gets explored without
# displacing a known-fast model.
UNKNOWN_MODEL_LATENCY_PRIOR = 3.0


def _record_latency(model: str, seconds: float) -> None:
    prev = _model_latency_ewma.get(model)
    if prev is None:
        _model_latency_ewma[model] = seconds
        return
    alpha = EWMA_ALPHA_WORSE if seconds > prev else EWMA_ALPHA_BETTER
    _model_latency_ewma[model] = alpha * seconds + (1 - alpha) * prev


def _candidate_models(candidate_models: list[str] | None) -> list[str]:
    """Order candidates fastest-measured-first, with unhealthy ones demoted to last.

    Sort is stable, so the caller's own ordering still breaks ties between models we
    have no measurement to separate.
    """
    raw = candidate_models or [
        settings.gemini_model,
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
    ]
    healthy = [m for m in raw if _is_model_healthy(m)]
    cooldown = [m for m in raw if not _is_model_healthy(m)]
    healthy.sort(key=lambda m: _model_latency_ewma.get(m, UNKNOWN_MODEL_LATENCY_PRIOR))
    return healthy + cooldown


# Verified live against the real API 2026-09-11: NOT every "3.x" model accepts a
# forced thinking_budget=0 — gemini-3.6-flash and gemini-3.5-flash-lite both reject
# it outright (400 INVALID_ARGUMENT). Only these two currently accept it, confirmed
# by direct trial. Left unset for everything else, those models fall back to the
# API's own default (implicit, non-zero) thinking budget — slower, but at least it
# doesn't hard-fail the call. This also fixed a real correctness bug: gemini-3.5-flash
# with no thinking_config was observed emitting truncated internal reasoning text
# ("No LaTeX? Yes. * No preamble/postamble? Yes...") as the visible answer when it
# ran out of max_output_tokens mid-thought — a raw chain-of-thought leak, which
# AGENTS.md explicitly forbids reaching an API response.
_THINKING_DISABLE_SUPPORTED = {"gemini-3.7-flash", "gemini-3.5-flash"}


def _build_config(
    model: str,
    temperature: float,
    max_output_tokens: int,
    system_instruction: str | None,
    response_mime_type: str | None,
) -> types.GenerateContentConfig:
    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        # We never pass `tools`, so there's nothing to auto-call — explicitly disabling
        # AFC (rather than leaving it at the SDK's enabled-by-default) silences its
        # "direct use of AFC is not recommended, use AsyncChat instead" warning, which
        # is about a code path we don't use.
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    if model in _THINKING_DISABLE_SUPPORTED:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=settings.gemini_thinking_budget
        )
    return types.GenerateContentConfig(**config_kwargs)


def _is_provider_error(exc: BaseException) -> bool:
    """True for a refusal from the provider, false for our own timeout.

    Only a provider refusal should trip the circuit breaker. Tripping it on our own
    TimeoutError locks out a healthy model — and because the cooldown is exponential,
    one impatient budget could sideline a working model for up to five minutes.

    Defined as "anything that is not our timeout" rather than by matching a list of
    status codes. An enumerated list is quietly wrong the first time a code outside it
    appears: a real 403 PERMISSION_DENIED was being logged as a local timeout and left
    the breaker untripped, so every request kept re-attempting all three models against
    a project that had been denied access outright.
    """
    return not isinstance(exc, (asyncio.TimeoutError, TimeoutError))


async def _attempt_generate(
    client: genai.Client,
    model: str,
    contents: Any,
    config: types.GenerateContentConfig,
    timeout: float,
) -> str:
    """One model call. Records its latency, raises on timeout/refusal/empty response."""
    started = time.monotonic()
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(model=model, contents=contents, config=config),
            timeout=timeout,
        )
    except BaseException:
        # A timeout, refusal, OR cancellation is still evidence about this model's
        # current speed — feeding it in demotes a stalling model on the next request
        # instead of letting it keep winning the ordering on a stale fast measurement.
        #
        # BaseException, not Exception, specifically because of cancellation:
        # asyncio.CancelledError derives from BaseException, and losing a hedge is
        # exactly how a chronically slow model exits. Catching only Exception meant the
        # one case we most need to learn from — "this model was still not done when a
        # rival finished" — was the one case that taught the router nothing, so it kept
        # being tried first and kept costing a hedge delay on every single request.
        _record_latency(model, time.monotonic() - started)
        raise
    text = response.text or ""
    if not text:
        _record_latency(model, time.monotonic() - started)
        raise RuntimeError(f"Model {model} returned an empty response")
    _record_latency(model, time.monotonic() - started)
    return text


def _note_failure(model: str, exc: BaseException, budget: float) -> None:
    provider_error = _is_provider_error(exc)
    if provider_error:
        _mark_model_unhealthy(model)
    else:
        _mark_model_timed_out(model)
    logger.warning(
        "Model call failed",
        attempted_model=model,
        # Empty-string errors were previously indistinguishable from provider
        # failures in the logs — TimeoutError stringifies to "". Name the cause.
        failure_kind="provider_error" if provider_error else "local_timeout",
        timeout_budget=round(budget, 2),
        error_summary=str(exc)[:120] or type(exc).__name__,
    )


# How many models may have a request in flight at the same time. Hedging trades a
# little extra quota for a large cut in tail latency; 2 is enough to cover one dead
# model without multiplying load across the whole chain.
HEDGE_MAX_IN_FLIGHT = 2


async def generate_gemini_content(
    contents: Any,
    system_instruction: str | None = None,
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
    response_mime_type: str | None = None,
    candidate_models: list[str] | None = None,
    timeout: float | None = None,
    hedge_delay: float | None = None,
) -> str:
    """Generate content, racing a second model when the first one stalls.

    Strict sequential fallback — try model A for its full timeout, then B, then C —
    makes every request pay for the slowest thing in the chain before it can reach a
    working model. Measured live 2026-09-11: gemini-3.6-flash (the configured primary)
    was answering 0 of 5 calls, so a synthesis request burned its entire 10s budget on
    a dead model before even starting on one that worked. Two of those in a row is most
    of a 20s answer, and none of it is generation.

    So a slow model no longer blocks the next one: after `hedge_delay` with nothing back,
    the next candidate starts IN PARALLEL and the first usable response wins. Latency
    becomes roughly `hedge_delay + (fastest live model)` instead of the sum of every
    preceding model's timeout. This is the standard hedged-request pattern (Dean &
    Barroso, "The Tail at Scale") and it is what makes a p99 look like a p50 when the
    backend's latency is as variable as this one's is.

    Uses the native async client — no thread-pool hop.
    """
    client = get_genai_client()
    if client is None:
        raise ValueError("GEMINI_API_KEY is not configured")

    models_to_try = _candidate_models(candidate_models)
    # Budgets sized from measured latency, not optimism: a grounded generation over ten
    # sources routinely runs past the old 5s/8s ceilings, and every attempt now gets the
    # full budget (see _is_provider_error for why the old 60% fallback budget backfired).
    base_timeout = timeout if timeout is not None else (20.0 if max_output_tokens > 512 else 12.0)
    # Long enough that a healthy model (measured 0.95-2.2s for classification) normally
    # answers before we ever hedge, short enough that a dead one can't eat the budget.
    hedge_after = hedge_delay if hedge_delay is not None else max(1.5, min(3.0, base_timeout / 3))

    deadline = time.monotonic() + base_timeout
    in_flight: dict[asyncio.Task[str], tuple[str, float]] = {}
    last_error: Exception | None = None
    next_idx = 0

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            # Start another model if one is available and we're under the hedge cap.
            #
            # A hedge (something is already running) is speculative, so it must not be
            # spent on a model we already know is failing. Observed live 2026-09-11:
            # with gemini-3.6-flash returning 503 and gemini-3.7-flash returning 429
            # RESOURCE_EXHAUSTED, every request was hedging onto both of them in turn —
            # burning the very quota whose exhaustion was causing the slowness, and
            # making the next request more likely to be throttled. Hedging is only
            # worth doing onto a model that might actually answer.
            #
            # A *fallback* (nothing left in flight) is different and still tries any
            # candidate, healthy or not: a known-bad model beats no attempt at all.
            if next_idx < len(models_to_try) and len(in_flight) < HEDGE_MAX_IN_FLIGHT:
                model = models_to_try[next_idx]
                is_speculative_hedge = len(in_flight) > 0
                if is_speculative_hedge and not _is_model_healthy(model):
                    next_idx += 1
                    continue
                next_idx += 1
                config = _build_config(
                    model, temperature, max_output_tokens, system_instruction, response_mime_type
                )
                task = asyncio.ensure_future(
                    _attempt_generate(client, model, contents, config, remaining)
                )
                in_flight[task] = (model, time.monotonic())
                if len(in_flight) > 1:
                    logger.info(
                        "Hedging request onto an additional model",
                        hedged_model=model,
                        waiting_on=[m for t, (m, _) in in_flight.items() if t is not task],
                        hedge_after=round(hedge_after, 2),
                    )

            if not in_flight:
                break

            # Wait only until the hedge window expires while candidates remain, so a
            # stalled model releases us to start the next one rather than holding the
            # whole request. Once everything is launched, wait out the real deadline.
            more_candidates = next_idx < len(models_to_try)
            wait_window = min(hedge_after if more_candidates else remaining, remaining)

            done, _pending = await asyncio.wait(
                set(in_flight), timeout=wait_window, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                model, _started = in_flight.pop(task)
                try:
                    text = task.result()
                except Exception as exc:
                    last_error = exc
                    _note_failure(model, exc, base_timeout)
                    continue

                _mark_model_healthy(model)
                logger.info("Model call succeeded", model=model, hedged=len(in_flight) > 0)
                return clean_json_response(text) if response_mime_type == "application/json" else text

            # Nothing finished inside the window, and nothing left to launch — the
            # remaining task(s) already hold the full deadline, so keep waiting.
    finally:
        # Losing hedges are pure waste from here on; don't leave them running.
        #
        # Their latency is recorded HERE, synchronously, rather than relying on the
        # cancelled coroutine to record its own: task.cancel() only *requests*
        # cancellation, and the task does not run its handler until the event loop
        # schedules it — which is after this function has already returned. The
        # loser's elapsed time is a lower bound on how slow it is, and a lower bound
        # is enough to demote it. Without this, the model that loses every race is
        # the one model the router never learns anything about.
        now = time.monotonic()
        for task, (model, started) in in_flight.items():
            if not task.done():
                _record_latency(model, now - started)
            task.cancel()

    if last_error:
        raise last_error
    raise RuntimeError("Failed to generate content with Gemini")


async def _open_stream(
    client: genai.Client,
    model: str,
    contents: Any,
    config: types.GenerateContentConfig,
    timeout: float,
) -> tuple[str, Any, str]:
    """Open a stream and pull its first non-empty chunk.

    Returns (model, stream, first_chunk). Waiting for the first chunk — rather than
    just for the stream object — is what makes this a real readiness signal: the SDK
    hands back a stream immediately, so a model that accepts the connection and then
    produces nothing would otherwise look like the winner of the race below.
    """
    started = time.monotonic()
    try:
        stream = await asyncio.wait_for(
            client.aio.models.generate_content_stream(model=model, contents=contents, config=config),
            timeout=timeout,
        )
        async for chunk in stream:
            text = chunk.text or ""
            if text:
                _record_latency(model, time.monotonic() - started)
                return model, stream, text
        raise RuntimeError(f"Model {model} produced an empty stream")
    except BaseException:
        # BaseException covers cancellation — see _attempt_generate for why that case
        # is the one the router most needs to learn from.
        _record_latency(model, time.monotonic() - started)
        raise


async def generate_gemini_content_stream(
    contents: Any,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1536,
    candidate_models: list[str] | None = None,
    timeout: float | None = None,
    hedge_delay: float | None = None,
) -> AsyncIterator[str]:
    """Stream content chunks as they're generated — true token-level streaming.

    Hedged on time-to-first-chunk only. This is the path the chat UI actually renders,
    so time-to-first-token *is* the latency the user experiences — and it was the thing
    a stalled model hurt most, since nothing at all appears on screen until some model
    starts producing. If the leading model hasn't emitted a first chunk within
    `hedge_delay`, the next candidate opens in parallel and whichever speaks first wins.

    Only the opening is raced. Once a stream wins we commit to it and never retry
    mid-flight, because by then its text is already on the user's screen — swapping
    models would rewrite an answer the user is in the middle of reading. Losing streams
    are cancelled as soon as a winner is picked, so the hedge costs at most a few
    duplicate opening tokens, never a duplicate full generation.

    No JSON mode: structured/streamed output don't mix, so streaming callers should use
    marker-based citations instead of response_mime_type="application/json".
    """
    client = get_genai_client()
    if client is None:
        raise ValueError("GEMINI_API_KEY is not configured")

    models_to_try = _candidate_models(candidate_models)
    # This budget covers only the wait for the stream to *open*, not the full
    # generation — once chunks are flowing the user is already seeing output.
    base_timeout = timeout if timeout is not None else 20.0
    hedge_after = hedge_delay if hedge_delay is not None else max(2.0, min(4.0, base_timeout / 4))

    deadline = time.monotonic() + base_timeout
    in_flight: dict[asyncio.Task[tuple[str, Any, str]], tuple[str, float]] = {}
    last_error: Exception | None = None
    next_idx = 0
    winner: tuple[str, Any, str] | None = None

    try:
        while winner is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            if next_idx < len(models_to_try) and len(in_flight) < HEDGE_MAX_IN_FLIGHT:
                model = models_to_try[next_idx]
                # Never spend a speculative hedge on a known-failing model — see the
                # non-streaming path above for the quota-burn this prevents.
                if len(in_flight) > 0 and not _is_model_healthy(model):
                    next_idx += 1
                    continue
                next_idx += 1
                config = _build_config(model, temperature, max_output_tokens, system_instruction, None)
                task = asyncio.ensure_future(
                    _open_stream(client, model, contents, config, remaining)
                )
                in_flight[task] = (model, time.monotonic())
                if len(in_flight) > 1:
                    logger.info("Hedging stream onto an additional model", hedged_model=model)

            if not in_flight:
                break

            more_candidates = next_idx < len(models_to_try)
            wait_window = min(hedge_after if more_candidates else remaining, remaining)

            done, _pending = await asyncio.wait(
                set(in_flight), timeout=wait_window, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                model, _started = in_flight.pop(task)
                try:
                    winner = task.result()
                    break
                except Exception as exc:
                    last_error = exc
                    _note_failure(model, exc, base_timeout)
    finally:
        # Record-then-cancel, for the same reason as the non-streaming path above.
        now = time.monotonic()
        for task, (model, started) in in_flight.items():
            if not task.done():
                _record_latency(model, now - started)
            task.cancel()

    if winner is None:
        if last_error:
            raise last_error
        raise RuntimeError("Failed to stream content with Gemini")

    won_by, stream, first_chunk = winner
    _mark_model_healthy(won_by)
    logger.info("Stream opened", model=won_by)
    yield first_chunk
    async for chunk in stream:
        text = chunk.text or ""
        if text:
            yield text
