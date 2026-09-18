"""Conversation and Message models for persistent multi-turn thread memory.

Fields:
- Conversation:
  - id: UUID primary key (thread_id)
  - user_id: Foreign key to User
  - title: Auto-generated or user-defined thread title
  - summary: Running compressed summary of every message older than the sliding window
  - message_count: total messages in the thread — avoids a COUNT(*) query every turn
  - summarized_through_count: how many of the oldest messages are already folded into
    `summary` — lets the memory service fold only the newly-evicted messages each turn
    instead of re-summarizing everything (see agents/memory/service.py)
  - created_at / updated_at: audit timestamps
- Message:
  - id: UUID primary key
  - conversation_id: Foreign key to Conversation (CASCADE)
  - role: 'user' | 'assistant' | 'system'
  - content: Full message text
  - raw_query: Original user prompt if rewritten
  - rewritten_query: Coreference-resolved search query if modified
  - citations: JSON list of citation items
  - created_at: message timestamp
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.database import Base


class Conversation(Base):
    """A persistent multi-turn conversation thread."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        default="New Conversation",
        nullable=False,
    )
    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    message_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
        nullable=False,
    )
    summarized_through_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default="0",
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:
        return f"<Conversation {self.id} title={self.title!r}>"


class Message(Base):
    """An individual message turn within a conversation thread."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(50),
        nullable=False,  # "user" | "assistant" | "system"
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    raw_query: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    rewritten_query: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation",
        back_populates="messages",
    )

    def __repr__(self) -> str:
        return f"<Message {self.id} role={self.role!r} conv={self.conversation_id}>"
