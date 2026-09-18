"""Direct LLM node — handles conversational greetings and non-retrieval chat.

When the Query Analyzer identifies intent="direct_llm" (e.g. "hi", "how are you",
"who created you"), this node generates a direct conversational response without
triggering database or web retrieval pipelines.
"""

from __future__ import annotations

import structlog

from agents.state import AgentState

logger = structlog.get_logger(__name__)

DIRECT_SYSTEM_PROMPT = """You are OmniMind, an enterprise AI assistant powered by advanced multi-modal Retrieval-Augmented Generation.
You are helpful, polite, concise, and professional.
Respond naturally to the user's conversational query."""


async def direct_llm_node(state: AgentState) -> AgentState:
    """Generate a direct conversational response without retrieval."""
    query = state.get("query", "").strip()
    route_history = list(state.get("route_history", []))
    route_history.append("direct_llm")

    token_callback = state.get("token_callback")

    # Fast offline responses for basic greetings
    lower = query.lower()
    if lower in ("hi", "hello", "hey", "greetings"):
        answer = "Hello! I am OmniMind, your enterprise AI assistant. How can I help you today?"
        if token_callback is not None:
            await token_callback(answer)
        return {
            **state,
            "draft_answer": answer,
            "final_answer": answer,
            "citations": [],
            "verification_status": "DIRECT_RESPONSE",
            "faithfulness_score": 1.0,
            "route_history": route_history,
        }

    try:
        # Already the bounded sliding window (agents/memory/service.py) — no re-slicing needed.
        chat_history = state.get("chat_history") or []
        prompt_content = query
        if chat_history:
            history_lines = []
            for msg in chat_history:
                r = "User" if msg.get("role") == "user" else "Assistant"
                c = msg.get("content", "").strip()
                if c:
                    history_lines.append(f"{r}: {c[:200]}")
            if history_lines:
                prompt_content = f"Recent Dialogue:\n" + "\n".join(history_lines) + f"\n\nUser: {query}"

        # Casual conversation, not technical synthesis — lite-first cuts latency/cost
        # on the fast-path without a noticeable quality tradeoff.
        fast_path_models = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash"]

        if token_callback is not None:
            from agents.llm_helper import generate_gemini_content_stream

            text_pieces: list[str] = []
            async for piece in generate_gemini_content_stream(
                contents=prompt_content,
                system_instruction=DIRECT_SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=256,
                candidate_models=fast_path_models,
            ):
                text_pieces.append(piece)
                await token_callback(piece)
            answer = "".join(text_pieces).strip()
        else:
            from agents.llm_helper import generate_gemini_content

            answer = (
                await generate_gemini_content(
                    contents=prompt_content,
                    system_instruction=DIRECT_SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=256,
                    candidate_models=fast_path_models,
                )
            ).strip()

        logger.info("Direct LLM response generated", query=query, length=len(answer))

        return {
            **state,
            "draft_answer": answer,
            "final_answer": answer,
            "citations": [],
            "verification_status": "DIRECT_RESPONSE",
            "faithfulness_score": 1.0,
            "route_history": route_history,
        }

    except Exception as e:
        logger.warning("Direct LLM call failed, using fallback greeting", error=str(e))
        answer = "Hello! I am OmniMind, your enterprise assistant. How can I help you today?"
        if token_callback is not None:
            await token_callback(answer)
        return {
            **state,
            "draft_answer": answer,
            "final_answer": answer,
            "citations": [],
            "verification_status": "DIRECT_RESPONSE",
            "faithfulness_score": 1.0,
            "route_history": route_history,
        }
