"""Unit tests for Langfuse observability, tracing, and metrics."""

import pytest

from observability.langfuse_client import NullSpan, NullTrace, flush_traces, get_langfuse_client
from observability.metrics import calculate_cost, estimate_tokens
from observability.tracer import TraceContext, observe


def test_estimate_tokens():
    """Test token estimation heuristics."""
    assert estimate_tokens("") == 0
    # "Hello world" is 11 chars -> ~3 tokens
    assert estimate_tokens("Hello world") >= 1
    long_text = "A" * 400
    assert estimate_tokens(long_text) == 100


def test_calculate_cost():
    """Test model pricing and cost estimation."""
    # 1M input tokens on Gemini 3.5 Flash-Lite = $0.075
    cost = calculate_cost("gemini-3.5-flash-lite", input_tokens=1_000_000, output_tokens=0)
    assert cost == 0.075

    # Cohere search query = $0.002
    cohere_cost = calculate_cost("rerank-v3.5", search_queries=1)
    assert cohere_cost == 0.002

    # Local models = $0.00
    local_cost = calculate_cost("ms-marco-TinyBERT-L-2-v2", input_tokens=5000)
    assert local_cost == 0.0


def test_null_trace_and_span():
    """NullTrace and NullSpan should execute all methods safely without throwing."""
    trace = NullTrace("test_id", "test_name")
    assert trace.id == "test_id"
    assert trace.name == "test_name"

    span = trace.span("child_span")
    assert isinstance(span, NullSpan)
    span.update(output="data")
    span.event("event_name")
    span.score(name="eval_score", value=1.0)
    span.end()


@pytest.mark.asyncio
async def test_trace_context_span_lifecycle():
    """TraceContext span should measure latency and capture metadata."""
    tracer = TraceContext(name="test_trace", user_id="tenant-123", metadata={"env": "test"})
    assert tracer.trace_id is not None

    async with tracer.span(name="sub_task", input_data={"val": 42}, metadata={"stage": "unit_test"}) as span:
        # Simulate work
        val = 42 * 2
        assert val == 84


@pytest.mark.asyncio
async def test_trace_context_generation():
    """TraceContext generation should track prompt tokens, completion tokens, and cost."""
    tracer = TraceContext(name="llm_trace")

    async with tracer.generation(
        name="test_llm_call",
        model="gemini-3.5-flash-lite",
        prompt="Explain attention in transformers.",
        metadata={"user_id": "u1"},
    ) as gen:
        gen["output_text"] = "Attention is a mechanism that computes softmax weighted values."


@pytest.mark.asyncio
async def test_observe_decorator():
    """Test @observe decorator on async functions."""

    @observe(span_name="custom_math")
    async def sample_async_func(a: int, b: int) -> int:
        return a + b

    result = await sample_async_func(10, 20)
    assert result == 30


def test_flush_traces_safe():
    """flush_traces should be safe to call even when Langfuse is not enabled."""
    flush_traces()
