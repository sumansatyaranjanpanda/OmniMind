"""Chat endpoint — the user-facing API for the Adaptive CRAG agent.

POST /chat accepts a user query and returns a citation-verified answer with full
provenance metadata. The server is the sole authority for conversation history: the
client sends only `query` (+ optional `thread_id`); history is loaded from and
persisted to Postgres via agents/memory/service.py. Clients never resend history.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from agents.graph import run_agent
from agents.memory.service import load_conversation_context, persist_turn_background
from api.database import get_db
from api.deps import get_current_user
from api.models.user import User

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Incoming chat request from the user."""

    query: str = Field(..., description="The user's natural language question.")
    thread_id: str | None = Field(
        None,
        description="Existing conversation thread ID. Omit to start a new thread.",
    )
    max_iterations: int = Field(
        2,
        ge=1,
        le=5,
        description="Maximum correction loops for the citation critic (circuit breaker).",
    )


class CitationItem(BaseModel):
    """A single citation linking a claim to its evidence source."""

    marker: str = ""
    chunk_id: str = ""
    text_snippet: str = ""
    source_type: str = "document"
    url: str | None = None
    page: int | None = None
    section: str | None = None


class ChatResponse(BaseModel):
    """Complete response from the Adaptive CRAG agent."""

    answer: str
    citations: list[CitationItem] = Field(default_factory=list)
    verification_status: str = Field(
        ...,
        description="VERIFIED | PARTIALLY_VERIFIED | INSUFFICIENT_EVIDENCE | CACHED | BLOCKED",
    )
    guardrail_status: str = Field(
        "PASSED",
        description="Guardrail evaluation status: PASSED | BLOCKED | PII_MASKED",
    )
    guardrail_reason: str | None = Field(
        None,
        description="Specific reason if a guardrail policy was triggered or entity masked.",
    )
    faithfulness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well the answer is grounded in source evidence (0.0–1.0).",
    )
    route_history: list[str] = Field(
        default_factory=list,
        description="Ordered list of graph nodes executed for this query.",
    )
    raw_query: str | None = Field(None, description="Original user query prior to coreference resolution.")
    rewritten_query: str | None = Field(None, description="Contextualized query if modified by rewriter.")
    thread_id: str | None = Field(None, description="Active conversation thread ID — pass this back on the next turn.")
    cache_hit: bool = Field(False, description="Whether the answer was served from semantic cache.")
    iteration_count: int = Field(0, description="Number of correction iterations performed.")
    intent: str = Field("internal_rag", description="Classified intent: internal_rag | web_search | hybrid | direct_llm | blocked")
    retrieval_trace_id: str | None = Field(
        None,
        description="Langfuse trace ID for the retrieval pipeline.",
    )


@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ChatResponse:
    """Ask OmniMind a question and get a citation-verified, hallucination-checked answer.

    The Adaptive Multi-Source pipeline executes:
    1. Load thread context (sliding window + rolling summary) from Postgres
    2. Input Guardrail safety check (Injection & PII Masking)
    3. Semantic cache check (Redis / In-memory)
    4. Contextual Query Rewriter (coreference resolution) + episodic memory recall, in parallel
    5. Query Analysis & Intent Classification (Direct | Internal | Web | Hybrid)
    6. Parallel Multi-Source Retrieval
    7. Source-Aware Evidence Fusion
    8. Context-Aware Citation-backed answer synthesis
    9. Faithfulness verification with Query Rewriter correction loops
    10. Output Guardrail credential sanitization
    11. Persist the turn (and fold evicted history into the rolling summary if needed)
    """
    tenant_id = str(current_user.id)

    ctx = await load_conversation_context(db, request.thread_id, current_user.id)

    result = await run_agent(
        query=request.query,
        tenant_id=tenant_id,
        user_id=str(current_user.id),
        chat_history=ctx.recent_messages,
        conversation_summary=ctx.summary,
        thread_id=str(ctx.conversation.id),
        max_iterations=request.max_iterations,
    )

    citations = [
        CitationItem(**c) if isinstance(c, dict) else c
        for c in result.get("citations", [])
    ]

    answer = result.get("answer", "")
    thread_id_str = str(ctx.conversation.id)

    # Persistence (and the occasional incremental summary-fold LLM call it can trigger)
    # is not on the request's critical path — it happens after the response is already
    # on the wire, same as /chat/stream. Uses its own DB session (see
    # persist_turn_background's docstring) since this request's session is torn down
    # by the time BackgroundTasks actually run.
    background_tasks.add_task(
        persist_turn_background,
        thread_id=thread_id_str,
        user_id=str(current_user.id),
        query=request.query,
        final_answer=answer,
        raw_query=result.get("raw_query"),
        rewritten_query=result.get("rewritten_query"),
        citations=[dict(c) for c in result.get("citations", [])],
    )

    return ChatResponse(
        answer=answer,
        citations=citations,
        verification_status=result.get("verification_status", ""),
        guardrail_status=result.get("guardrail_status", "PASSED"),
        guardrail_reason=result.get("guardrail_reason"),
        faithfulness_score=result.get("faithfulness_score", 0.0),
        route_history=result.get("route_history", []),
        intent=result.get("intent", "internal_rag"),
        raw_query=result.get("raw_query", request.query),
        rewritten_query=result.get("rewritten_query"),
        thread_id=thread_id_str,
        cache_hit=result.get("cache_hit", False),
        iteration_count=result.get("iteration_count", 0),
        retrieval_trace_id=result.get("retrieval_trace_id"),
    )
