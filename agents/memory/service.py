"""Conversation memory persistence — sliding window + rolling summary, per thread.

Design: a bounded window of the last MEMORY_WINDOW_MESSAGES messages is loaded verbatim
on every turn; everything older only ever exists as a rolling `summary` on the
Conversation row. When a turn pushes messages out of the window, exactly those
newly-evicted messages are folded into the summary with one small LLM call —
never a re-summarization of history that's already been folded in. That incremental
fold (via `summarized_through_count`) is what keeps this O(1) per turn instead of
O(n) over the life of a conversation.

Episodic (cross-thread) recall is a separate concern — see retrieval/memory_store.py.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.conversation import Conversation, Message

logger = structlog.get_logger(__name__)

MEMORY_WINDOW_MESSAGES = 8  # last 4 user/assistant turn-pairs, kept verbatim

FOLD_SYSTEM_PROMPT = """You are a concise Conversation Summarizer.
Merge the existing summary with the newly evicted turns into one updated summary of
2-4 dense bullet points. Preserve specific technical names, versions, numbers, and
user preferences. Do not restate anything already covered by the existing summary —
only add what the new turns contribute."""


@dataclass
class ConversationContext:
    """Everything a request needs to reason about a thread's history."""

    conversation: Conversation
    recent_messages: list[dict[str, str]] = field(default_factory=list)
    summary: str | None = None


async def load_conversation_context(
    db: AsyncSession, thread_id: str | None, user_id: uuid.UUID
) -> ConversationContext:
    """Load (or create) a conversation and its sliding window + rolling summary."""
    conversation: Conversation | None = None

    if thread_id:
        try:
            result = await db.execute(
                select(Conversation).where(
                    Conversation.id == uuid.UUID(thread_id),
                    Conversation.user_id == user_id,
                )
            )
            conversation = result.scalar_one_or_none()
        except ValueError:
            conversation = None  # malformed UUID from a stale client — start a fresh thread

    if conversation is None:
        conversation = Conversation(user_id=user_id)
        db.add(conversation)
        await db.flush()

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc())
        .limit(MEMORY_WINDOW_MESSAGES)
    )
    recent = list(reversed(result.scalars().all()))
    recent_messages = [{"role": m.role, "content": m.content} for m in recent]

    return ConversationContext(
        conversation=conversation,
        recent_messages=recent_messages,
        summary=conversation.summary,
    )


async def _fold_evicted_into_summary(existing_summary: str | None, evicted: list[Message]) -> str:
    """Fold newly-evicted messages into the existing summary with ONE incremental LLM call."""
    if not evicted:
        return existing_summary or ""

    try:
        from agents.llm_helper import generate_gemini_content

        evicted_lines = [
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content[:300]}" for m in evicted
        ]
        prompt = (
            f"Existing summary:\n{existing_summary or '(none yet)'}\n\n"
            "Newly evicted turns to fold in:\n" + "\n".join(evicted_lines) +
            "\n\nUpdated summary:"
        )
        summary = await generate_gemini_content(
            contents=prompt,
            system_instruction=FOLD_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=256,
            candidate_models=["gemini-3.5-flash-lite", "gemini-3.6-flash"],
            timeout=4.0,
        )
        return summary.strip()
    except Exception as exc:
        logger.warning("Summary fold failed; keeping prior summary (non-critical)", error=str(exc))
        return existing_summary or ""


async def persist_turn(
    db: AsyncSession,
    conversation: Conversation,
    user_content: str,
    assistant_content: str,
    raw_query: str | None = None,
    rewritten_query: str | None = None,
    citations: list[dict] | None = None,
) -> tuple[Message, Message]:
    """Save one user/assistant turn; fold newly-evicted messages into the summary if needed."""
    # Explicit timestamps: two inserts in the same transaction would otherwise share
    # Postgres's now() (transaction-start time), making user/assistant ordering ambiguous.
    turn_time = datetime.now(timezone.utc)

    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=user_content,
        raw_query=raw_query,
        rewritten_query=rewritten_query,
        created_at=turn_time,
    )
    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=assistant_content,
        citations=citations,
        created_at=turn_time + timedelta(microseconds=1),
    )
    db.add_all([user_msg, assistant_msg])

    if conversation.message_count == 0 and conversation.title == "New Conversation":
        conversation.title = user_content[:80]

    total_before = conversation.message_count
    total_after = total_before + 2
    already_folded = conversation.summarized_through_count

    newly_evicted_count = max(0, total_after - MEMORY_WINDOW_MESSAGES) - max(
        0, total_before - MEMORY_WINDOW_MESSAGES
    )

    if newly_evicted_count > 0:
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.asc())
            .offset(already_folded)
            .limit(newly_evicted_count)
        )
        evicted = list(result.scalars().all())
        conversation.summary = await _fold_evicted_into_summary(conversation.summary, evicted)
        conversation.summarized_through_count = already_folded + len(evicted)

    conversation.message_count = total_after
    await db.flush()

    return user_msg, assistant_msg


async def persist_turn_background(
    thread_id: str,
    user_id: str,
    query: str,
    final_answer: str,
    raw_query: str | None,
    rewritten_query: str | None,
    citations: list[dict],
) -> None:
    """Persist a turn (and embed it for episodic recall) outside the request path.

    Shared by both /chat and /chat/stream. Opens its own DB session rather than
    reusing the route's request-scoped one, since that session is committed and torn
    down as soon as the endpoint function returns — before a FastAPI BackgroundTask
    or a post-stream continuation actually runs. Never raises: a failed persist means
    this turn falls out of memory for future turns, which is degraded, not broken.
    """
    import asyncio

    from api.database import async_session_factory
    from retrieval.memory_store import upsert_memory_turn

    try:
        async with async_session_factory() as db:
            result = await db.execute(
                select(Conversation).where(Conversation.id == uuid.UUID(thread_id))
            )
            conversation = result.scalar_one_or_none()
            if conversation is None:
                return

            user_msg, assistant_msg = await persist_turn(
                db,
                conversation,
                user_content=query,
                assistant_content=final_answer,
                raw_query=raw_query,
                rewritten_query=rewritten_query,
                citations=citations,
            )
            await db.commit()

        asyncio.create_task(
            upsert_memory_turn(user_id=user_id, thread_id=thread_id, message_id=str(user_msg.id), role="user", content=query)
        )
        asyncio.create_task(
            upsert_memory_turn(user_id=user_id, thread_id=thread_id, message_id=str(assistant_msg.id), role="assistant", content=final_answer)
        )
    except Exception as exc:
        logger.warning("Background turn persistence failed (non-critical)", error=str(exc))
