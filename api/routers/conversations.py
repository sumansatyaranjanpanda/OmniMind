"""Conversations router — list threads and rehydrate a thread's message history.

Exists so the frontend can survive a browser refresh: the server is the sole
authority for chat history (see chat.py), so the client needs a way to fetch it
back after reloading rather than resending it on every turn.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.deps import get_current_user
from api.models.conversation import Conversation, Message
from api.models.user import User

router = APIRouter(prefix="/conversations", tags=["conversations"])


class ConversationSummary(BaseModel):
    thread_id: str
    title: str
    updated_at: str
    message_count: int


class MessageItem(BaseModel):
    role: str
    content: str
    citations: list[dict] | None = None
    created_at: str


class ConversationDetail(BaseModel):
    thread_id: str
    title: str
    summary: str | None
    messages: list[MessageItem]


@router.get("", response_model=list[ConversationSummary])
async def list_conversations(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ConversationSummary]:
    """List the current user's conversation threads, most recently active first."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(50)
    )
    conversations = result.scalars().all()
    return [
        ConversationSummary(
            thread_id=str(c.id),
            title=c.title,
            updated_at=c.updated_at.isoformat(),
            message_count=c.message_count,
        )
        for c in conversations
    ]


@router.get("/{thread_id}", response_model=ConversationDetail)
async def get_conversation(
    thread_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ConversationDetail:
    """Fetch full message history for one thread — used to rehydrate the UI after a refresh."""
    try:
        thread_uuid = uuid.UUID(thread_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    result = await db.execute(
        select(Conversation).where(
            Conversation.id == thread_uuid, Conversation.user_id == current_user.id
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == thread_uuid)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()

    return ConversationDetail(
        thread_id=str(conversation.id),
        title=conversation.title,
        summary=conversation.summary,
        messages=[
            MessageItem(
                role=m.role,
                content=m.content,
                citations=m.citations,
                created_at=m.created_at.isoformat(),
            )
            for m in messages
        ],
    )
