"""Contextual Query Rewriter node — multi-turn coreference resolution + episodic recall.

Two independent jobs, run concurrently so neither adds latency to the other:
1. Resolve pronouns/ellipsis in follow-up queries ("How much does it cost?") into a
   standalone query, using the sliding window + rolling summary that
   agents/memory/service.py already loaded from Postgres. This node never
   summarizes anything itself — that bookkeeping lives entirely in the memory
   service so it only ever happens once, on write, not on every read.
2. Surface semantically related turns from the user's OTHER conversation threads
   (episodic memory) — independent of whether this query is a follow-up, since
   cross-thread recall is useful even in a brand-new thread.
"""

from __future__ import annotations

import asyncio
import json

import structlog

from agents.state import AgentState

logger = structlog.get_logger(__name__)

CONTEXT_REWRITER_SYSTEM_PROMPT = """You are an expert Contextual Query Rewriter for an enterprise Multi-Modal RAG system.

Given the recent conversation history and the user's latest follow-up query:
1. Determine if the query depends on prior context (contains pronouns like "it", "they", "that", "this", ellipsis, or implicit follow-ups).
2. If it is a context-dependent follow-up, rewrite it into a crisp, standalone, fully self-contained question that can be understood in isolation by search engines and vector retrievers.
3. If the query is already self-contained, a standalone question, or a casual greeting ("hi", "hello", "thanks"), return the original query unchanged.
4. Never answer the question. Only rewrite the query.

Output ONLY valid JSON matching this schema:
{
  "is_follow_up": true | false,
  "rewritten_query": "<self-contained query>",
  "resolved_references": "<brief explanation of what was resolved or 'none'>"
}"""


async def _rewrite_query(
    query: str, chat_history: list[dict[str, str]], conversation_summary: str | None
) -> dict | None:
    """One LLM call: resolve coreferences against the window + summary. None on failure."""
    history_blocks: list[str] = []
    if conversation_summary:
        history_blocks.append(f"[Prior Conversation Summary]:\n{conversation_summary}")
    for msg in chat_history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            history_blocks.append(f"{role}: {content[:250]}")

    rewrite_prompt = (
        f"Conversation History:\n{chr(10).join(history_blocks)}\n\n"
        f"Latest User Query: {query}\n\n"
        f"Rewrite into a standalone query if needed (JSON format):"
    )

    try:
        from agents.llm_helper import generate_gemini_content

        raw_json = await generate_gemini_content(
            contents=rewrite_prompt,
            system_instruction=CONTEXT_REWRITER_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=256,
            response_mime_type="application/json",
            candidate_models=["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash"],
            timeout=10.0,
        )
        if raw_json.startswith("```"):
            raw_json = raw_json.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw_json)
    except Exception as exc:
        logger.warning("Context rewriter failed; proceeding with original query", error=str(exc))
        return None


async def _retrieve_episodic(query: str, user_id: str, thread_id: str | None) -> list[dict]:
    """Cross-thread semantic recall. Already exception-safe internally — never raises."""
    from retrieval.memory_store import retrieve_episodic_memories

    return await retrieve_episodic_memories(query, user_id=user_id, exclude_thread_id=thread_id)


async def context_rewriter_node(state: AgentState) -> AgentState:
    """Resolve follow-up queries and surface cross-thread episodic memories, in parallel."""
    query = state.get("query", "").strip()
    raw_query = state.get("raw_query") or query
    chat_history = state.get("chat_history") or []
    conversation_summary = state.get("conversation_summary")
    thread_id = state.get("thread_id")
    user_id = state.get("user_id", "default")
    route_history = list(state.get("route_history", []))
    route_history.append("context_rewriter")

    episodic_task = asyncio.ensure_future(_retrieve_episodic(query, user_id, thread_id))

    # Fast path: no window means nothing to rewrite against — but episodic recall
    # (other threads) is still worth checking, so we still await that task below.
    if not chat_history:
        episodic_memories = await episodic_task
        return {
            **state,
            "query": query,
            "raw_query": raw_query,
            "rewritten_query": None,
            "conversation_summary": conversation_summary,
            "episodic_memories": episodic_memories,
            "route_history": route_history,
        }

    rewrite_result, episodic_memories = await asyncio.gather(
        _rewrite_query(query, chat_history, conversation_summary), episodic_task
    )

    if rewrite_result:
        is_follow_up = rewrite_result.get("is_follow_up", False)
        rewritten = (rewrite_result.get("rewritten_query") or "").strip()
        resolved_refs = rewrite_result.get("resolved_references", "none")

        if is_follow_up and rewritten and rewritten.lower() != query.lower():
            logger.info(
                "Coreference resolved: Query rewritten for retrieval",
                original_query=query,
                rewritten_query=rewritten,
                resolved_refs=resolved_refs,
            )
            return {
                **state,
                "query": rewritten,
                "raw_query": raw_query,
                "rewritten_query": rewritten,
                "conversation_summary": conversation_summary,
                "episodic_memories": episodic_memories,
                "route_history": route_history,
            }

    return {
        **state,
        "query": query,
        "raw_query": raw_query,
        "rewritten_query": None,
        "conversation_summary": conversation_summary,
        "episodic_memories": episodic_memories,
        "route_history": route_history,
    }
