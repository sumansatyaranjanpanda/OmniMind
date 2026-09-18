"""Token counting, latency profiling, and cost calculation for OmniMind models."""

from __future__ import annotations

import math
from typing import Any

# Pricing per 1,000,000 tokens in USD
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-3.7-flash": {"input": 0.10, "output": 0.40},
    "gemini-3.5-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "models/gemini-embedding-2": {"input": 0.00002, "output": 0.0},
    "gemini-embedding-2": {"input": 0.00002, "output": 0.0},
    "rerank-v3.5": {"per_search": 0.002},
    "rerank-english-v3.0": {"per_search": 0.002},
    "jinaai/jina-clip-v2": {"input": 0.0, "output": 0.0},
    "ms-marco-tinybert-l-2-v2": {"input": 0.0, "output": 0.0},
    "flashrank": {"input": 0.0, "output": 0.0},
}


def estimate_tokens(text: str) -> int:
    """Fast character-based heuristic token estimator (~4 chars per token)."""
    if not text:
        return 0
    # Standard rule of thumb for English & Markdown text: ~4 characters per token
    return max(1, math.ceil(len(text) / 4.0))


def calculate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    search_queries: int = 0,
) -> float:
    """Calculate the estimated USD cost of an operation."""
    pricing = MODEL_PRICING.get(model.lower(), {"input": 0.10, "output": 0.30})

    if "per_search" in pricing:
        return round(pricing["per_search"] * max(1, search_queries), 6)

    input_cost = (input_tokens / 1_000_000.0) * pricing.get("input", 0.0)
    output_cost = (output_tokens / 1_000_000.0) * pricing.get("output", 0.0)
    return round(input_cost + output_cost, 6)
