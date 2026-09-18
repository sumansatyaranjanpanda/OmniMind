"""LangGraph nodes for Input & Output Guardrails in OmniMind.

- input_guardrail_node: Evaluates incoming queries for injections, jailbreaks, and PII.
- output_guardrail_node: Scrubs secrets and leaked credentials from synthesized answers.
- finalize_blocked_node: Terminal node for policy-violating requests ($0 LLM tokens, <5ms exit).
"""

from __future__ import annotations

import structlog

from agents.state import AgentState
from core.guardrails import get_input_guardrail, get_output_guardrail

logger = structlog.get_logger(__name__)


async def input_guardrail_node(state: AgentState) -> AgentState:
    """Evaluate input query for prompt injections, malicious exploits, and PII.

    Executes deterministically in <5ms without calling external LLM APIs.
    """
    route_history = list(state.get("route_history", []))
    route_history.append("input_guardrail")

    query = state.get("query", "")
    guardrail = get_input_guardrail()
    result = guardrail.validate_query(query)

    logger.info(
        "Input guardrail evaluated query",
        status=result.status,
        reason=result.reason,
        has_pii=bool(result.pii_entities),
    )

    if not result.is_valid:
        refusal = result.refusal_message or (
            "I am unable to fulfill this request because it conflicts with our safety policies "
            "regarding system instruction overrides. If you are exploring technical concepts "
            "or have a related question, please feel free to rephrase your query."
        )
        return {
            **state,
            "guardrail_status": "BLOCKED",
            "guardrail_reason": result.reason,
            "draft_answer": refusal,
            "final_answer": refusal,
            "verification_status": "BLOCKED",
            "faithfulness_score": 0.0,
            "route_history": route_history,
        }

    # If PII was detected and masked, update query to masked query for downstream RAG
    effective_query = result.masked_text if result.status == "PII_MASKED" else query

    return {
        **state,
        "query": effective_query,
        "masked_query": result.masked_text if result.status == "PII_MASKED" else None,
        "pii_mapping": result.pii_mapping,
        "guardrail_status": result.status,
        "guardrail_reason": result.reason,
        "route_history": route_history,
    }


async def output_guardrail_node(state: AgentState) -> AgentState:
    """Scrub leaked credentials, API keys, or private secrets from the answer."""
    route_history = list(state.get("route_history", []))
    route_history.append("output_guardrail")

    answer = state.get("final_answer") or state.get("draft_answer", "")
    output_guardrail = get_output_guardrail()
    sanitized_answer, was_sanitized = output_guardrail.sanitize_output(answer)

    if was_sanitized:
        logger.warning("Output guardrail sanitized credentials from response")

    return {
        **state,
        "draft_answer": sanitized_answer,
        "final_answer": sanitized_answer,
        "route_history": route_history,
    }


async def finalize_blocked_node(state: AgentState) -> AgentState:
    """Terminal node for safety violations — exits immediately with zero LLM tokens."""
    route_history = list(state.get("route_history", []))
    route_history.append("finalize_blocked")

    return {
        **state,
        "verification_status": "BLOCKED",
        "faithfulness_score": 0.0,
        "route_history": route_history,
    }
