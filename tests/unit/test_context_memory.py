"""Unit tests for OmniMind's memory architecture.

Covers three layers:
- agents/memory/service.py   — sliding window + incremental (not quadratic) summary folding
- agents/nodes/context_rewriter.py — coreference resolution + parallel episodic recall
- retrieval/memory_store.py  — cross-thread episodic embedding/retrieval (local fallback)
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.nodes.context_rewriter import context_rewriter_node
from agents.nodes.direct_llm import direct_llm_node
from agents.nodes.synthesizer import synthesizer_node
from agents.state import AgentState
from api.models.conversation import Conversation, Message


# ── Context Rewriter Tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_rewriter_empty_history_still_checks_episodic_memory():
    """Empty window skips the rewrite LLM call, but episodic recall still runs."""
    state: AgentState = {
        "query": "What is Pinecone?",
        "chat_history": [],
        "user_id": "user-1",
        "thread_id": "thread-1",
        "route_history": ["input_guardrail", "cache_check"],
    }

    with patch(
        "agents.nodes.context_rewriter._retrieve_episodic",
        AsyncMock(return_value=[{"role": "user", "text": "prior unrelated note", "score": 0.8}]),
    ) as mock_episodic:
        result = await context_rewriter_node(state)

    mock_episodic.assert_awaited_once()
    assert result["query"] == "What is Pinecone?"
    assert result["rewritten_query"] is None
    assert result["episodic_memories"] == [{"role": "user", "text": "prior unrelated note", "score": 0.8}]
    assert "context_rewriter" in result["route_history"]


@pytest.mark.asyncio
async def test_context_rewriter_resolves_pronoun_followup():
    """Follow-up queries with pronouns are rewritten into standalone self-contained queries."""
    state: AgentState = {
        "query": "How much does it cost?",
        "chat_history": [
            {"role": "user", "content": "What is Pinecone vector database?"},
            {"role": "assistant", "content": "Pinecone is a managed serverless vector database designed for high-performance RAG and similarity search."},
        ],
        "user_id": "user-1",
        "thread_id": "thread-1",
        "route_history": ["cache_check"],
    }

    mock_llm_response = '{"is_follow_up": true, "rewritten_query": "What is the pricing and cost of Pinecone vector database?", "resolved_references": "it -> Pinecone vector database"}'

    with patch("agents.llm_helper.generate_gemini_content", AsyncMock(return_value=mock_llm_response)), \
         patch("agents.nodes.context_rewriter._retrieve_episodic", AsyncMock(return_value=[])):
        result = await context_rewriter_node(state)

    assert result["raw_query"] == "How much does it cost?"
    assert result["query"] == "What is the pricing and cost of Pinecone vector database?"
    assert result["rewritten_query"] == "What is the pricing and cost of Pinecone vector database?"
    assert "context_rewriter" in result["route_history"]


@pytest.mark.asyncio
async def test_context_rewriter_preserves_standalone_query():
    """If query is already standalone, rewritten_query remains None."""
    state: AgentState = {
        "query": "What is the capital of France?",
        "chat_history": [
            {"role": "user", "content": "Tell me about Redis."},
            {"role": "assistant", "content": "Redis is an in-memory data structure store."},
        ],
        "user_id": "user-1",
        "thread_id": "thread-1",
        "route_history": ["cache_check"],
    }

    mock_llm_response = '{"is_follow_up": false, "rewritten_query": "What is the capital of France?", "resolved_references": "none"}'

    with patch("agents.llm_helper.generate_gemini_content", AsyncMock(return_value=mock_llm_response)), \
         patch("agents.nodes.context_rewriter._retrieve_episodic", AsyncMock(return_value=[])):
        result = await context_rewriter_node(state)

    assert result["query"] == "What is the capital of France?"
    assert result["rewritten_query"] is None


@pytest.mark.asyncio
async def test_context_rewriter_runs_rewrite_and_episodic_concurrently():
    """Rewrite LLM call and episodic retrieval must run via asyncio.gather, not sequentially."""
    state: AgentState = {
        "query": "How much does it cost?",
        "chat_history": [{"role": "user", "content": "Tell me about Pinecone."}],
        "user_id": "user-1",
        "thread_id": "thread-1",
        "route_history": [],
    }

    with patch("agents.llm_helper.generate_gemini_content", AsyncMock(return_value='{"is_follow_up": false, "rewritten_query": "How much does it cost?", "resolved_references": "none"}')), \
         patch("agents.nodes.context_rewriter._retrieve_episodic", AsyncMock(return_value=[])) as mock_episodic:
        await context_rewriter_node(state)

    mock_episodic.assert_awaited_once_with("How much does it cost?", "user-1", "thread-1")


# ── Context-Aware Synthesizer Tests ─────────────────────────────────


@pytest.mark.asyncio
async def test_synthesizer_injects_window_summary_and_episodic_memory():
    """Synthesizer formats window + summary + episodic memories into three distinct prompt sections."""
    captured_prompts = []

    async def mock_generate(contents, **kwargs):
        captured_prompts.append(contents)
        return '{"answer": "Pinecone serverless pricing starts with a free tier. [^1]", "citations_used": [1]}'

    state: AgentState = {
        "query": "What is the pricing of Pinecone vector database?",
        "raw_query": "How much does it cost?",
        "chat_history": [
            {"role": "user", "content": "Tell me about Pinecone."},
            {"role": "assistant", "content": "Pinecone is a managed serverless vector database."},
        ],
        "conversation_summary": "- User explored Pinecone vector database architecture",
        "episodic_memories": [
            {"role": "user", "text": "I previously asked about Weaviate pricing too", "score": 0.81}
        ],
        "fused_evidence": [
            {
                "source_id": 1,
                "marker": "[^1]",
                "source_type": "document",
                "title": "Pricing Guide",
                "content": "Pinecone serverless pricing is usage-based.",
            }
        ],
        "route_history": ["retriever", "source_fusion"],
    }

    with patch("agents.llm_helper.generate_gemini_content", side_effect=mock_generate):
        result = await synthesizer_node(state)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "Prior Conversation Summary" in prompt
    assert "Tell me about Pinecone" in prompt
    assert "Relevant memory from earlier conversations" in prompt
    assert "Weaviate pricing" in prompt
    assert "never cite this as evidence" in prompt
    assert "Contextualized from original question: 'How much does it cost?'" in prompt
    assert result["draft_answer"] != ""


# ── Direct LLM Multi-Turn Tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_direct_llm_multi_turn_dialogue():
    """Direct LLM receives previous dialogue turns for coherent conversational chat."""
    captured_contents = []

    async def mock_generate(contents, **kwargs):
        captured_contents.append(contents)
        return "Nice to continue our conversation, Alice!"

    state: AgentState = {
        "query": "What was my name again?",
        "chat_history": [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice!"},
        ],
        "route_history": [],
    }

    with patch("agents.llm_helper.generate_gemini_content", side_effect=mock_generate):
        result = await direct_llm_node(state)

    assert len(captured_contents) == 1
    assert "Alice" in captured_contents[0]
    assert result["final_answer"] == "Nice to continue our conversation, Alice!"


# ── Database Models Tests ───────────────────────────────────────────


def test_conversation_and_message_models():
    """SQLAlchemy models for Conversation and Message can be instantiated with relationships."""
    user_id = uuid.uuid4()
    conv = Conversation(
        id=uuid.uuid4(),
        user_id=user_id,
        title="Vector Database Research",
        summary="User discussed Pinecone and Matryoshka embeddings.",
    )

    msg1 = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="user",
        content="What is Pinecone?",
        raw_query="What is Pinecone?",
    )

    msg2 = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="assistant",
        content="Pinecone is a managed vector database. [^1]",
        citations=[{"marker": "[^1]", "chunk_id": "chunk-1"}],
    )

    assert conv.title == "Vector Database Research"
    assert msg1.role == "user"
    assert msg2.citations[0]["marker"] == "[^1]"
    assert "Vector Database Research" in repr(conv)
    assert "assistant" in repr(msg2)


# ── Memory Service: Persistence + Incremental Summary Folding ──────


def _fake_db(evicted_messages: list[Message] | None = None) -> MagicMock:
    """Minimal async-session stub covering only what persist_turn/load_conversation_context use."""
    db = MagicMock()
    db.add = MagicMock()
    db.add_all = MagicMock()
    db.flush = AsyncMock()

    result = MagicMock()
    result.scalars.return_value.all.return_value = evicted_messages or []
    result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=result)
    return db


def _new_conversation(**overrides) -> Conversation:
    """A Conversation as it would look right after load_conversation_context's flush()."""
    defaults = dict(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        title="New Conversation",
        summary=None,
        message_count=0,
        summarized_through_count=0,
    )
    defaults.update(overrides)
    return Conversation(**defaults)


@pytest.mark.asyncio
async def test_persist_turn_first_message_sets_title_no_eviction():
    from agents.memory.service import persist_turn

    conversation = _new_conversation()
    db = _fake_db()

    with patch("agents.memory.service._fold_evicted_into_summary", AsyncMock()) as mock_fold:
        await persist_turn(db, conversation, user_content="What is Pinecone?", assistant_content="It's a vector DB.")

    mock_fold.assert_not_called()
    db.execute.assert_not_awaited()  # no eviction fetch needed under the window
    assert conversation.title == "What is Pinecone?"
    assert conversation.message_count == 2
    assert conversation.summarized_through_count == 0


@pytest.mark.asyncio
async def test_persist_turn_folds_only_newly_evicted_messages_not_full_history():
    """Regression test for the original bug: summarization must be O(1) per turn, not O(n).

    A conversation already at the window boundary (8 messages, 0 already folded) receives
    one more turn (2 messages). Exactly the 2 oldest messages should be fetched and folded —
    never the full history, and never messages already covered by summarized_through_count.
    """
    from agents.memory.service import persist_turn

    conversation = _new_conversation(
        message_count=8, summarized_through_count=0, summary="- Some prior summary"
    )
    evicted = [
        Message(id=uuid.uuid4(), conversation_id=conversation.id, role="user", content="oldest turn"),
        Message(id=uuid.uuid4(), conversation_id=conversation.id, role="assistant", content="oldest reply"),
    ]
    db = _fake_db(evicted_messages=evicted)

    with patch(
        "agents.memory.service._fold_evicted_into_summary", AsyncMock(return_value="- Updated summary")
    ) as mock_fold:
        await persist_turn(db, conversation, user_content="new question", assistant_content="new answer")

    mock_fold.assert_awaited_once()
    fold_args = mock_fold.await_args.args
    assert fold_args[0] == "- Some prior summary"
    assert len(fold_args[1]) == 2  # only the newly-evicted pair, never the full history
    assert conversation.summary == "- Updated summary"
    assert conversation.summarized_through_count == 2  # advances by exactly what was folded
    assert conversation.message_count == 10


@pytest.mark.asyncio
async def test_persist_turn_already_overflowing_evicts_exactly_two_more():
    """Once folding is underway (already_folded > 0), each turn evicts exactly the 2 new messages."""
    from agents.memory.service import persist_turn

    conversation = _new_conversation(
        message_count=12, summarized_through_count=4, summary="- Existing rolling summary"
    )
    evicted = [
        Message(id=uuid.uuid4(), conversation_id=conversation.id, role="user", content="turn 5 user"),
        Message(id=uuid.uuid4(), conversation_id=conversation.id, role="assistant", content="turn 5 assistant"),
    ]
    db = _fake_db(evicted_messages=evicted)

    with patch(
        "agents.memory.service._fold_evicted_into_summary", AsyncMock(return_value="- Extended summary")
    ) as mock_fold:
        await persist_turn(db, conversation, user_content="q", assistant_content="a")

    assert len(mock_fold.await_args.args[1]) == 2
    assert conversation.summarized_through_count == 6
    assert conversation.message_count == 14


@pytest.mark.asyncio
async def test_fold_evicted_into_summary_uses_existing_summary_as_context():
    """The fold prompt must include the existing summary, not just the evicted turns."""
    from agents.memory.service import _fold_evicted_into_summary

    evicted = [Message(id=uuid.uuid4(), conversation_id=uuid.uuid4(), role="user", content="I use Postgres")]

    captured = []

    async def mock_generate(contents, **kwargs):
        captured.append(contents)
        return "- User uses Postgres"

    with patch("agents.llm_helper.generate_gemini_content", side_effect=mock_generate):
        summary = await _fold_evicted_into_summary("- User is researching databases", evicted)

    assert "User is researching databases" in captured[0]
    assert "I use Postgres" in captured[0]
    assert summary == "- User uses Postgres"


@pytest.mark.asyncio
async def test_load_conversation_context_creates_new_thread_when_none_given():
    from agents.memory.service import load_conversation_context

    db = _fake_db()
    user_id = uuid.uuid4()

    ctx = await load_conversation_context(db, thread_id=None, user_id=user_id)

    db.add.assert_called_once()
    assert ctx.conversation.user_id == user_id
    assert ctx.recent_messages == []
    assert ctx.summary is None


@pytest.mark.asyncio
async def test_load_conversation_context_loads_existing_thread():
    from agents.memory.service import load_conversation_context

    existing = _new_conversation(summary="- prior summary", message_count=2)
    messages = [
        Message(id=uuid.uuid4(), conversation_id=existing.id, role="user", content="hi"),
        Message(id=uuid.uuid4(), conversation_id=existing.id, role="assistant", content="hello"),
    ]

    db = MagicMock()
    conv_result = MagicMock()
    conv_result.scalar_one_or_none.return_value = existing
    msg_result = MagicMock()
    msg_result.scalars.return_value.all.return_value = list(reversed(messages))  # DB returns desc order
    db.execute = AsyncMock(side_effect=[conv_result, msg_result])

    ctx = await load_conversation_context(db, thread_id=str(existing.id), user_id=existing.user_id)

    assert ctx.conversation is existing
    assert ctx.summary == "- prior summary"
    assert ctx.recent_messages == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


# ── Episodic Memory Store (Stage B) ─────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_memory_turn_local_fallback_stores_vector():
    from retrieval import memory_store

    fake_store: dict[str, list[dict]] = {}

    with patch("retrieval.memory_store.get_async_index", return_value=None), \
         patch.object(memory_store, "_local_vector_store", fake_store), \
         patch("retrieval.memory_store.generate_text_embeddings", AsyncMock(return_value=[[0.1, 0.2, 0.3]])):
        await memory_store.upsert_memory_turn(
            user_id="user-1", thread_id="thread-1", message_id="msg-1", role="user", content="I use Postgres"
        )

    assert "memory-user-1" in fake_store
    assert fake_store["memory-user-1"][0]["metadata"]["thread_id"] == "thread-1"


@pytest.mark.asyncio
async def test_retrieve_episodic_memories_excludes_current_thread_and_low_scores():
    from retrieval import memory_store

    fake_store = {
        "memory-user-1": [
            {"id": "a", "values": [1.0, 0.0], "metadata": {"thread_id": "other-thread", "role": "user", "text": "relevant"}},
            {"id": "b", "values": [1.0, 0.0], "metadata": {"thread_id": "current-thread", "role": "user", "text": "same thread, excluded"}},
            {"id": "c", "values": [0.0, 1.0], "metadata": {"thread_id": "other-thread", "role": "user", "text": "unrelated, low score"}},
        ]
    }

    with patch("retrieval.memory_store.get_async_index", return_value=None), \
         patch.object(memory_store, "_local_vector_store", fake_store), \
         patch("retrieval.memory_store.generate_text_embeddings", AsyncMock(return_value=[[1.0, 0.0]])):
        results = await memory_store.retrieve_episodic_memories(
            query="something relevant", user_id="user-1", exclude_thread_id="current-thread"
        )

    assert len(results) == 1
    assert results[0]["text"] == "relevant"
