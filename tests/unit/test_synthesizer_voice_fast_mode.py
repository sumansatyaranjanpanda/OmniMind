"""Regression guard for voice-mode synthesizer latency shaping.

History worth keeping, because it is the reason this file no longer asserts a model
name. The original 2026-09-10 fix hard-coded `gemini-3.5-flash-lite` first for voice,
on the strength of a live benchmark showing it 100% reliable at 1.4-1.7s while the two
flagships ahead of it failed ~50% of the time. Re-measuring on 2026-09-11 — interleaved,
to control for drift — showed that same model at 10.9-17.0s in one run and 0.95s twenty
minutes later, and showed the configured primary answering 0 of 5 calls.

Provider speed here moves by ~10x within the hour, so a pinned order is a bug with a
delay fuse. Model *selection* now lives in agents/llm_helper.py, which measures latency
per call and orders candidates from that data. What stays voice-specific, and is what
these tests pin, is the shape of the request: a short draft and a tight budget, because
the Live model compresses the result into 2-3 spoken sentences regardless.
"""

from unittest.mock import AsyncMock, patch

import pytest

from agents.nodes.synthesizer import synthesizer_node


def _state(**overrides):
    base = {
        "query": "what does the SLA say about uptime",
        "fused_evidence": [
            {
                "source_id": 1,
                "marker": "[^1]",
                "source_type": "document",
                "title": "SLA",
                "content": "Uptime guarantee is 99.9%.",
            }
        ],
        "route_history": [],
        "chat_history": [],
        "token_callback": None,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_typed_chat_keeps_quality_first_defaults():
    """No voice_fast_mode set -> full-length draft and the standard budget."""
    with patch(
        "agents.llm_helper.generate_gemini_content",
        new_callable=AsyncMock,
        return_value="Uptime is 99.9%. [^1]",
    ) as mock:
        await synthesizer_node(_state())

    kwargs = mock.await_args.kwargs
    assert kwargs["max_output_tokens"] == 1536
    assert kwargs["timeout"] == 10.0


@pytest.mark.asyncio
async def test_voice_fast_mode_shortens_the_draft_and_the_budget():
    with patch(
        "agents.llm_helper.generate_gemini_content",
        new_callable=AsyncMock,
        return_value="Uptime is 99.9%. [^1]",
    ) as mock:
        await synthesizer_node(_state(voice_fast_mode=True))

    kwargs = mock.await_args.kwargs
    assert kwargs["max_output_tokens"] < 1536
    assert kwargs["timeout"] < 10.0


@pytest.mark.asyncio
async def test_voice_fast_mode_does_not_pin_a_model_list():
    """Model choice belongs to the latency-aware router, not to this node.

    Pinning it here is what went stale within a day last time; if someone
    reintroduces a hard-coded chain, this fails and points them at the reason.
    """
    with patch(
        "agents.llm_helper.generate_gemini_content",
        new_callable=AsyncMock,
        return_value="Uptime is 99.9%. [^1]",
    ) as mock:
        await synthesizer_node(_state(voice_fast_mode=True))

    assert "candidate_models" not in mock.await_args.kwargs


@pytest.mark.asyncio
async def test_voice_fast_mode_still_produces_a_cited_answer():
    """The speed trade must not silently drop citation extraction or gating."""
    with patch(
        "agents.llm_helper.generate_gemini_content",
        new_callable=AsyncMock,
        return_value="The uptime guarantee is 99.9%. [^1]",
    ):
        result = await synthesizer_node(_state(voice_fast_mode=True))

    assert result["citations"]
    assert result["citations"][0]["marker"] == "[^1]"
