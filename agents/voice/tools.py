"""Tools the Live model calls mid-conversation to ground what it says.

Two tools, and the model's choice between them is the whole routing layer —
there is no separate router LLM, which is why routing costs nothing here:

``search_documents``  Tier 1. One hybrid retrieval pass, no query rewrite, no
                      synthesis LLM. The Live model reads the returned evidence
                      and speaks from it. Target ~1s.

``deep_research``     Tier 2. Hands the question to the full text agent graph —
                      critic, faithfulness gate, rewrite loop and all — and the
                      Live model just voices the verified result. Slower (~3-5s),
                      but the answer is exactly as rigorous as a typed one. This
                      is the reason voice mode does not have to trade quality
                      away for latency; it only has to *cover* the latency with
                      speech, which the model does by acknowledging first.

The evidence floor is the safety mechanism that replaces the inline critic.
Faithfulness scoring is post-hoc by nature — it grades an answer after it
exists — and a spoken sentence cannot be retracted. Most faithfulness failures
trace back to thin or absent evidence rather than to a model inventing things
on top of good evidence, so the check moves upstream: if nothing clears the
floor, the tool returns ``status="no_evidence"`` and the system instruction
requires the model to say it has nothing rather than improvise.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog

from api.config import get_settings
from core.guardrails import sanitize_retrieved_content
from retrieval.models import RetrievedChunk
from retrieval.pipeline import execute_retrieval

logger = structlog.get_logger(__name__)


# Spoken answers quote one or two sources, so the evidence block handed to the
# model is trimmed hard. Sending 5 full chunks wastes audio-model context and
# measurably slows first-token time without improving what gets said.
_MAX_SNIPPET_CHARS = 900


def build_function_declarations() -> list[dict[str, Any]]:
    """Function declarations advertised to the Live model.

    Plain dicts rather than ``types.FunctionDeclaration`` so this stays
    importable (and unit-testable) without constructing a genai client.
    """
    return [
        {
            "name": "search_documents",
            "description": (
                "Search the user's own uploaded documents and speak the answer from "
                "what comes back. Use this for almost every factual question — it is "
                "fast. Always call it before answering anything about the user's "
                "files, projects, data, or people mentioned in their documents."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": (
                            "The search query, with pronouns already resolved from the "
                            "conversation. If the user asks 'what about his role?' after "
                            "discussing Abinash, search 'Abinash role', not 'his role'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "deep_research",
            "description": (
                "Full verified research pass over the user's documents, with citation "
                "checking. Slower — takes a few seconds — so tell the user you're "
                "looking into it before calling. Use ONLY for questions that need "
                "comparison, synthesis across several documents, multi-step reasoning, "
                "or where being wrong would be costly. Never use it for simple lookups."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "The full question, with pronouns resolved.",
                    }
                },
                "required": ["query"],
            },
        },
    ]


def _confidence(chunk: RetrievedChunk) -> float | None:
    """Best comparable relevance score for a chunk, or None if it has none.

    Deliberately never falls back to ``rrf_score``. RRF values are reciprocal
    ranks (~0.016 for a top hit with k=60), so comparing one against a floor
    calibrated for cosine/rerank scores would reject every chunk ever retrieved.
    Returning None means "cannot judge", and the caller lets those through
    rather than silently discarding real evidence.
    """
    if chunk.rerank_score is not None:
        return float(chunk.rerank_score)
    if chunk.dense_score is not None:
        return float(chunk.dense_score)
    return None


# Verified live 2026-09-09: the same underlying fact scored 0.457 (passes the 0.35
# floor) for the query "Abinash" and 0.259 (rejected) for "Who is Abhinash" — the
# Live model's own natural phrasing of the identical question, compounded by the
# STT mishearing the name. A cross-encoder scores a terse entity-name query against
# a resume chunk much higher than a full interrogative sentence, so ordinary phrasing
# variance from the Live model — which is not deterministic — can flip a correct
# answer into a false "not in your documents". This regex strips exactly that kind
# of question-wrapper without an LLM round-trip, so a rejected first attempt gets one
# cheap second try before voice actually gives up on a fact that's really there.
_QUESTION_PREFIX = re.compile(
    r"^(?:can you |could you |do you know |please )?"
    r"(?:tell me about|who(?:'s| is)|what(?:'s| is)|where(?:'s| is)|"
    r"when(?:'s| is)|why is|how is)\s+",
    re.IGNORECASE,
)


def _normalize_query(query: str) -> str | None:
    """Strip a leading question-wrapper, e.g. 'Who is Abinash?' -> 'Abinash'.

    Returns None when normalization wouldn't change anything, so the caller
    can skip a pointless identical retry.
    """
    stripped = _QUESTION_PREFIX.sub("", query).rstrip("? ").strip()
    if not stripped or stripped.lower() == query.strip().lower():
        return None
    return stripped


def _source_label(chunk: RetrievedChunk) -> str:
    """Human-speakable source name — the model reads this aloud as attribution."""
    meta = chunk.metadata or {}
    name = meta.get("filename") or meta.get("source") or meta.get("doc_id") or "a document"
    section = meta.get("heading") or meta.get("section")
    page = meta.get("page_number") or meta.get("page")
    if section:
        return f"{name}, section '{section}'"
    if page:
        return f"{name}, page {page}"
    return str(name)


class VoiceToolbox:
    """Tool implementations bound to one authenticated voice session.

    Bound per-session rather than taking ``tenant_id`` as a tool argument on
    purpose: the Live model must never be able to influence which tenant's
    corpus is searched. The model supplies only the query text; the tenant
    comes from the validated JWT on the WebSocket.
    """

    def __init__(self, tenant_id: str, user_id: str, thread_id: str) -> None:
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.thread_id = thread_id
        # Citations accrue across the session so the transcript pane can show
        # sources even though speech never reads chunk IDs out loud.
        self.citations: list[dict[str, Any]] = []
        self._citation_index = 0

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Route a Live function call to its implementation.

        Never raises: a tool exception inside the Live receive loop would tear
        down the audio session mid-sentence. Failures come back as a spoken-
        friendly payload instead so the conversation survives them.
        """
        query = str(args.get("query", "")).strip()
        if not query:
            return {"status": "error", "message": "No search query was provided."}

        try:
            if name == "search_documents":
                return await self.search_documents(query)
            if name == "deep_research":
                return await self.deep_research(query)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Voice tool failed", tool=name, error=str(exc))
            return {
                "status": "error",
                "message": "The document search failed. Tell the user you hit a technical problem.",
            }

        return {"status": "error", "message": f"Unknown tool '{name}'."}

    async def search_documents(self, query: str) -> dict[str, Any]:
        """Tier 1 — one hybrid retrieval pass, no LLM in the loop.

        On a rejected first attempt, retries once with the question-wrapper
        stripped (see ``_normalize_query``) before reporting no evidence. This
        exists because of a verified live failure: the Live model's own choice
        of phrasing is not deterministic, and a full question sentence can score
        below the floor for a chunk that a terse entity-name query clears easily
        — for the same fact, in the same document. One cheap retry closes that
        gap without adding an LLM round-trip to the common case.
        """
        result = await self._search_once(query)
        if result["status"] == "no_evidence":
            normalized = _normalize_query(query)
            if normalized is not None:
                logger.info("Voice search retrying with normalized query", original=query, normalized=normalized)
                retry = await self._search_once(normalized)
                if retry["status"] == "ok":
                    return retry
        return result

    async def _search_once(self, query: str) -> dict[str, Any]:
        settings = get_settings()

        chunks, trace_id = await execute_retrieval(
            query=query,
            tenant_id=self.tenant_id,
            top_k=settings.voice_retrieval_top_k,
            # Off: costs a full LLM round-trip, and the Live model already sends
            # a well-formed, coreference-resolved query per the tool description.
            enable_query_rewrite=False,
            # On: BM25 + RRF is local and fast. Turning it off would cost recall
            # for no meaningful latency win — the round-trips were never here.
            enable_hybrid=True,
            enable_rerank=settings.voice_enable_rerank,
        )

        if not chunks:
            return {
                "status": "no_evidence",
                "message": (
                    "Nothing in the user's documents matches this. Say plainly that you "
                    "could not find it in their documents. Do not answer from general "
                    "knowledge and do not guess."
                ),
            }

        floor = settings.voice_evidence_floor
        kept: list[RetrievedChunk] = []
        for chunk in chunks:
            score = _confidence(chunk)
            # None = unscored, which means "cannot judge" rather than "bad" —
            # dropping those would throw away evidence on the local fallback
            # path, where no reranker or cosine score is attached at all.
            if score is None or score >= floor:
                kept.append(chunk)

        if not kept:
            logger.info(
                "Voice evidence gate rejected all chunks",
                query=query,
                floor=floor,
                candidates=len(chunks),
                best=max((_confidence(c) or 0.0) for c in chunks),
            )
            return {
                "status": "no_evidence",
                "message": (
                    "The documents contain nothing relevant enough to answer this. Say "
                    "you could not find it in their documents. Do not guess."
                ),
            }

        evidence: list[dict[str, Any]] = []
        for chunk in kept:
            # Document text is untrusted input — the same delimiter-injection payload
            # that gets BLOCKED when a user types it would otherwise reach the model
            # verbatim through a retrieved chunk. Same boundary the text lane enforces
            # in source_fusion; voice does not get an exemption from it.
            safe_text, was_flagged = sanitize_retrieved_content(chunk.text)
            if was_flagged:
                logger.warning("Sanitized injection attempt in voice evidence", chunk_id=chunk.id)

            self._citation_index += 1
            label = _source_label(chunk)
            meta = chunk.metadata or {}

            evidence.append(
                {
                    "source": label,
                    "content": safe_text[:_MAX_SNIPPET_CHARS],
                }
            )
            self.citations.append(
                {
                    "marker": f"[^{self._citation_index}]",
                    "chunk_id": chunk.id,
                    "text_snippet": safe_text[:400],
                    "source_type": "document",
                    "url": None,
                    "page": meta.get("page_number") or meta.get("page"),
                    "section": meta.get("heading") or meta.get("section"),
                }
            )

        logger.info(
            "Voice search_documents succeeded",
            query=query,
            kept=len(kept),
            trace_id=trace_id,
        )
        return {
            "status": "ok",
            "evidence": evidence,
            "instruction": (
                "Answer from this evidence only, in one or two spoken sentences. "
                "Name the source out loud once (for example 'according to your résumé'). "
                "If the evidence does not actually cover the question, say so instead."
            ),
        }

    async def deep_research(self, query: str) -> dict[str, Any]:
        """Tier 2 — the full text agent graph, critic and faithfulness gate included."""
        from agents.graph import get_compiled_graph
        from agents.state import AgentState

        initial_state: AgentState = {
            "query": query,
            "raw_query": query,
            "rewritten_query": None,
            "thread_id": self.thread_id,
            "conversation_summary": None,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "chat_history": [],
            "episodic_memories": [],
            "token_callback": None,
            "voice_fast_mode": True,
            "iteration": 0,
            "max_iterations": 2,
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

        compiled = get_compiled_graph()
        final_state = await compiled.ainvoke(initial_state)

        answer = final_state.get("final_answer") or final_state.get("draft_answer") or ""
        if not answer:
            return {
                "status": "no_evidence",
                "message": "The research pass produced no answer. Tell the user you could not find this.",
            }

        for citation in final_state.get("citations", []):
            self.citations.append(dict(citation) if not isinstance(citation, dict) else citation)

        logger.info(
            "Voice deep_research completed",
            query=query,
            faithfulness=final_state.get("faithfulness_score"),
            verification=final_state.get("verification_status"),
        )
        return {
            "status": "ok",
            "answer": answer,
            "verification_status": final_state.get("verification_status", "VERIFIED"),
            "instruction": (
                "This answer has already been fact-checked against the documents. "
                "Speak it back conversationally and briefly — summarize it in two or "
                "three sentences rather than reading it out verbatim. Do not read "
                "citation markers like [^1] out loud."
            ),
        }
