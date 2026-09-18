"""Shared state definition for the OmniMind Adaptive Multi-Source Agent graph.

The AgentState is the single source of truth that flows through every node
in the LangGraph state machine. Each node reads from and writes to this
TypedDict, enabling fully traceable, deterministic decision-making.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, TypedDict

from retrieval.models import RetrievedChunk

TokenCallback = Callable[[str], Awaitable[None]]


class Citation(TypedDict, total=False):
    """A single verified citation linking a claim to its source."""

    marker: str  # e.g. "[^1]"
    chunk_id: str
    text_snippet: str  # The exact passage supporting this claim
    source_type: str  # "document" | "web"
    url: str | None  # Only for web citations
    page: int | None
    section: str | None


class SubQuery(TypedDict, total=False):
    """A decomposed sub-query with its assigned information source."""

    query: str
    source: str  # "internal" | "web"
    reasoning: str


class FusedEvidence(TypedDict, total=False):
    """Unified evidence item combining document chunks and web results with provenance."""

    source_id: int  # 1-indexed (1, 2, 3...)
    marker: str  # "[^1]", "[^2]"
    source_type: str  # "document" | "web"
    title: str
    content: str
    url: str | None
    doc_id: str | None
    chunk_id: str | None
    page: int | None
    section: str | None
    score: float | None


class AgentState(TypedDict, total=False):
    """Full state flowing through the Adaptive CRAG graph.

    Every field is optional (total=False) so nodes only write what they own.
    """

    # ── User Input & Contextual Memory ─────────────────────────
    query: str
    raw_query: str  # Original query before coreference rewrite
    rewritten_query: str | None  # Contextualized query if modified
    thread_id: str | None  # Conversation thread identifier
    conversation_summary: str | None  # Compressed summary of older turns
    user_id: str
    tenant_id: str
    chat_history: list[dict[str, str]]  # sliding window, already bounded by the memory service
    episodic_memories: list[dict[str, Any]]  # cross-thread recall, see retrieval/memory_store.py
    token_callback: TokenCallback | None  # set by /chat/stream for live token-level streaming;
    # absent (None) for the non-streaming /chat path, which just gets the blocking full-text call

    # Set only by agents/voice/tools.py's deep_research(), never by typed chat. Verified live
    # 2026-09-10: gemini-3.6-flash/3.7-flash were both returning 503 "high demand" ~50% of the
    # time, and gemini-3.5-flash — third in the default fallback chain — measured a CONSISTENT
    # 10.9-16.8s even when healthy, longer than the 10s timeout guarding it, so it was reliably
    # timed out and never actually got to answer. Every real deep_research call was burning
    # 20-30s thrashing through two overloaded flagships and one mistimed fallback before ever
    # reaching gemini-3.5-flash-lite, which measured 100% reliable at 1.4-1.7s. A voice user is
    # live on the line for every one of those seconds, unlike a typed-chat user — so this field
    # lets synthesizer.py bias toward speed for voice specifically, without touching the
    # quality-first default ordering that typed chat still gets.
    voice_fast_mode: bool

    # ── Guardrail Layer ─────────────────────────────────────────
    guardrail_status: str  # "PASSED" | "BLOCKED" | "PII_MASKED"
    guardrail_reason: str | None
    masked_query: str | None
    pii_mapping: dict[str, str]

    # ── Cache Layer ─────────────────────────────────────────────
    cache_hit: bool
    cached_answer: str | None

    # ── Query Analysis & Planning ───────────────────────────────
    intent: str  # "internal_rag" | "web_search" | "hybrid" | "direct_llm"
    complexity: str  # "simple" | "multi_hop"
    sub_queries: list[SubQuery]
    is_direct_response: bool

    # ── Retrieval Layer ─────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]
    retrieval_trace_id: str | None
    # Exactly which internal queries the speculative retrieval in
    # agents/graph.py:analyze_and_retrieve_node already searched, so retriever_node can
    # tell "this work is already done" from "this is a different search". Compared by
    # value rather than trusted as a boolean flag on purpose: the critic's retry loop
    # and the query rewriter both re-enter the retriever with DIFFERENT queries, and a
    # plain "already retrieved" flag would hand them the first search's stale chunks.
    speculative_retrieval_queries: list[str] | None
    graph_paths: list[dict[str, Any]]  # Subgraphs/entity relationship paths from GraphRAG

    # ── Web Augmentation ────────────────────────────────────────
    web_results: list[dict[str, Any]]  # [{title, url, snippet}]
    # True when the web was searched only because the user's own documents returned
    # nothing usable. The synthesizer says so out loud in that case: answering a
    # question about the user's corpus purely from public web pages, without
    # flagging it, is how "who is abinash" came back describing four strangers.
    web_is_fallback: bool
    # True when the web fallback fired because retrieval itself FAILED (Pinecone
    # unreachable, etc.), not because the corpus was searched and had nothing. Confirmed
    # live: a simulated Pinecone outage produced the same "I could not find this in your
    # documents" message as a genuinely empty search — false in the outage case, since
    # the documents were never actually checked. The two need different wording.
    retrieval_errored: bool

    # ── Corpus Awareness ────────────────────────────────────────
    # Filenames of the documents this tenant has actually ingested. Fed to the query
    # analyzer so routing is grounded in what the corpus contains instead of guessed
    # from the wording of the question alone.
    corpus_documents: list[str]

    # ── Fusion Layer ────────────────────────────────────────────
    fused_evidence: list[FusedEvidence]

    # ── Synthesis Layer ─────────────────────────────────────────
    draft_answer: str
    citations: list[Citation]
    # True when the model was unreachable and draft_answer is an apology rather than a
    # generated answer. Lets the critic skip verifying text no model produced.
    synthesis_failed: bool

    # ── Critic / Verification Layer ─────────────────────────────
    faithfulness_score: float  # 0.0–1.0
    verification_status: str  # "VERIFIED" | "PARTIALLY_VERIFIED" | "INSUFFICIENT_EVIDENCE"
    critic_feedback: str  # Explanation of what failed verification
    unsupported_claims: list[str]

    # ── Final Output ────────────────────────────────────────────
    final_answer: str
    route_history: list[str]  # ["cache_check", "query_analyzer", "retriever", ...]

    # ── Control Flow ────────────────────────────────────────────
    iteration: int  # Current correction loop count
    max_iterations: int  # Circuit breaker (default: 2)
    error: str | None  # If any node fails
