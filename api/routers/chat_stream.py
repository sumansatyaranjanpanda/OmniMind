"""Streaming Chat endpoint — Server-Sent Events (SSE) for real-time tokens & state events.

POST /chat/stream emits live event stream:
- event: "route_update" (shows agent progressing through cache -> analyzer -> retrieval)
- event: "citations" (delivers verified inline citations)
- event: "token" (streams words as the LLM actually generates them)
- event: "complete" (final metadata, faithfulness score, verification status)

Tokens are real: the synthesizer/direct_llm nodes stream directly from Gemini via a
callback threaded through agent state, pushing each chunk into a shared queue as it
arrives. A separate graph-progression loop pushes route/citation events into the same
queue. This generator just drains that one queue — so token events and route events
interleave in true wall-clock order instead of the route events waiting for a node to
fully finish, and instead of fake per-word `sleep()` calls after the fact.

Same server-authoritative memory model as POST /chat (see chat.py): the client sends
only `query` + optional `thread_id`. Context is loaded before streaming starts; the
turn is persisted after the "complete" event has already been sent, so persistence
never adds latency to what the user sees.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any, AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from agents.memory.service import load_conversation_context, persist_turn_background
from api.database import get_db
from api.deps import get_current_user
from api.models.user import User
from api.routers.chat import ChatRequest

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["streaming"])


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def generate_chat_events(
    query: str,
    tenant_id: str,
    user_id: str,
    thread_id: str,
    chat_history: list[dict[str, str]],
    conversation_summary: str | None,
    max_iterations: int = 2,
) -> AsyncGenerator[str, None]:
    """Drive the agent graph in the background and yield SSE events as they occur."""
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def emit(event: str, data: dict[str, Any]) -> None:
        await queue.put(_sse(event, data))

    async def token_callback(text: str) -> None:
        await emit("token", {"text": text})

    async def run_graph() -> None:
        try:
            from agents.graph import get_compiled_graph
            from agents.state import AgentState

            compiled = get_compiled_graph()
            initial_state: AgentState = {
                "query": query,
                "raw_query": query,
                "rewritten_query": None,
                "thread_id": thread_id,
                "conversation_summary": conversation_summary,
                "user_id": user_id or tenant_id,
                "tenant_id": tenant_id,
                "chat_history": chat_history or [],
                "episodic_memories": [],
                "token_callback": token_callback,
                "iteration": 0,
                "max_iterations": max_iterations,
                "route_history": [],
                "guardrail_status": "PASSED",
                "guardrail_reason": None,
                "masked_query": None,
                "pii_mapping": {},
                "cache_hit": False,
                "intent": "",
                "complexity": "simple",
                "sub_queries": [],
                "is_direct_response": False,
                "retrieved_chunks": [],
                "graph_paths": [],
                "web_results": [],
                "fused_evidence": [],
                "citations": [],
                "faithfulness_score": 0.0,
                "verification_status": "",
                "critic_feedback": "",
                "unsupported_claims": [],
                "draft_answer": "",
                "final_answer": "",
                "error": None,
            }

            await emit(
                "route_update",
                {
                    "status": "STARTING",
                    "message": "Evaluating input guardrails and analyzing intent...",
                    "route_history": ["input_guardrail"],
                },
            )

            accumulated_state: dict[str, Any] = dict(initial_state)

            async for chunk in compiled.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    accumulated_state.update(node_output)
                    route_history = accumulated_state.get("route_history", [])

                    if node_name == "input_guardrail":
                        g_status = node_output.get("guardrail_status", "PASSED")
                        g_reason = node_output.get("guardrail_reason")
                        await emit("guardrail_check", {"status": g_status, "reason": g_reason})
                        await emit("route_update", {"status": "GUARDRAIL_EVALUATED", "route_history": route_history})

                    elif node_name == "finalize_blocked":
                        await emit("route_update", {"status": "BLOCKED", "route_history": route_history})
                        ans = node_output.get("final_answer", "")
                        if ans:
                            await emit("token", {"text": ans})

                    elif node_name == "cache_check":
                        c_hit = node_output.get("cache_hit", False)
                        await emit(
                            "route_update",
                            {
                                "status": "CACHE_HIT" if c_hit else "CACHE_MISS",
                                "cache_hit": c_hit,
                                "route_history": route_history,
                            },
                        )

                    elif node_name == "finalize_cached":
                        # Cache hits are the fast path (~50ms) — emit the whole answer
                        # at once rather than re-adding artificial per-word delay.
                        c_ans = node_output.get("final_answer") or node_output.get("cached_answer", "")
                        citations = node_output.get("citations", [])
                        if citations:
                            await emit("citations", {"citations": citations})
                        if c_ans:
                            await emit("token", {"text": c_ans})

                    elif node_name == "context_rewriter":
                        rewritten = node_output.get("rewritten_query")
                        await emit(
                            "route_update",
                            {"status": "CONTEXT_REWRITTEN", "rewritten_query": rewritten, "route_history": route_history},
                        )

                    elif node_name == "query_analyzer":
                        intent = node_output.get("intent", "internal_rag")
                        await emit("route_update", {"status": "ANALYZED", "intent": intent, "route_history": route_history})

                    elif node_name in ("retriever", "graph_retriever", "web_search", "hybrid_multi_engine"):
                        await emit("route_update", {"status": "RETRIEVED", "route_history": route_history})

                    elif node_name == "source_fusion":
                        await emit("route_update", {"status": "FUSED", "route_history": route_history})

                    elif node_name in ("synthesizer", "direct_llm"):
                        # Tokens were already streamed live via token_callback while this
                        # node was running — only the citation list is new information here.
                        citations = node_output.get("citations", [])
                        if citations:
                            await emit("citations", {"citations": citations})

                    elif node_name == "critic":
                        score = node_output.get("faithfulness_score", 1.0)
                        v_status = node_output.get("verification_status", "VERIFIED")
                        await emit(
                            "route_update",
                            {
                                "status": "CRITIC_VERIFIED",
                                "faithfulness_score": score,
                                "verification_status": v_status,
                                "route_history": route_history,
                            },
                        )

                    elif node_name == "output_guardrail":
                        await emit("route_update", {"status": "OUTPUT_CHECKED", "route_history": route_history})

            guardrail_status = accumulated_state.get("guardrail_status", "PASSED")
            guardrail_reason = accumulated_state.get("guardrail_reason")
            final_answer = accumulated_state.get("final_answer") or accumulated_state.get("draft_answer", "")
            final_citations = accumulated_state.get("citations", [])

            await emit(
                "complete",
                {
                    "answer": final_answer,
                    "citations": final_citations,
                    "verification_status": accumulated_state.get("verification_status", "VERIFIED"),
                    "guardrail_status": guardrail_status,
                    "guardrail_reason": guardrail_reason,
                    "faithfulness_score": accumulated_state.get("faithfulness_score", 1.0),
                    "cache_hit": accumulated_state.get("cache_hit", False),
                    "iteration_count": accumulated_state.get("iteration", 0),
                    "route_history": accumulated_state.get("route_history", []),
                    "intent": accumulated_state.get("intent", "internal_rag"),
                    "raw_query": accumulated_state.get("raw_query", query),
                    "rewritten_query": accumulated_state.get("rewritten_query"),
                    "thread_id": thread_id,
                    "retrieval_trace_id": accumulated_state.get("retrieval_trace_id"),
                },
            )

            # Persistence happens after the client already has everything it needs.
            if guardrail_status != "BLOCKED":
                await persist_turn_background(
                    thread_id=thread_id,
                    user_id=user_id,
                    query=query,
                    final_answer=final_answer,
                    raw_query=accumulated_state.get("raw_query", query),
                    rewritten_query=accumulated_state.get("rewritten_query"),
                    citations=[dict(c) if not isinstance(c, dict) else c for c in final_citations],
                )

        except Exception as e:
            logger.error("Streaming chat failed", error=str(e))
            await emit("error", {"error": str(e)})
        finally:
            await queue.put(None)  # sentinel — tells the consumer loop below to stop

    graph_task = asyncio.create_task(run_graph())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        if not graph_task.done():
            graph_task.cancel()


@router.post("/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    """Stream token-by-token responses and live agent routing events via SSE."""
    tenant_id = str(current_user.id)
    user_id = str(current_user.id)

    ctx = await load_conversation_context(db, request.thread_id, current_user.id)

    event_generator = generate_chat_events(
        query=request.query,
        tenant_id=tenant_id,
        user_id=user_id,
        thread_id=str(ctx.conversation.id),
        chat_history=ctx.recent_messages,
        conversation_summary=ctx.summary,
        max_iterations=request.max_iterations,
    )

    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
