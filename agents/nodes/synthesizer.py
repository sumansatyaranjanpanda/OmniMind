"""Synthesizer node — generates draft answer with inline citations.

Takes fused multi-source evidence (document chunks and live web results) and produces
a structured answer where every factual claim is backed by a [^N] citation
marker linking to a specific source chunk or web URL.

Plain markdown output with inline [^N] markers — not JSON. Structured (JSON-mode)
output and token-level streaming don't mix: a client can't safely render partial JSON
as it arrives. Which sources got cited is derived by scanning the finished answer text
for [^N] markers instead of asking the model to report it separately, so the citation
list is always exactly consistent with what the answer actually references.
"""

from __future__ import annotations

import re

import structlog

from agents.state import AgentState, Citation, FusedEvidence

logger = structlog.get_logger(__name__)

# Matches one [^N] bracket AND tolerates the model occasionally grouping numbers into a
# single bracket (e.g. "[^3, ^4]" or "[^3,4]") — findall inside the bracket picks up every
# digit run so citation attribution stays correct even when the format rule isn't followed.
CITATION_MARKER_RE = re.compile(r"\[\^[\d,\s\^]+\]")
CITATION_NUMBER_RE = re.compile(r"\d+")

SYNTHESIS_SYSTEM_PROMPT = """You are an expert research assistant for the OmniMind RAG system.

Your task: Generate a comprehensive, accurate answer using ONLY the provided context.

CRITICAL RULES:
1. Every factual claim MUST have an inline citation marker [^N] where N corresponds to the source number.
2. Each bracket cites exactly ONE source. To cite multiple sources for one claim, write
   separate adjacent brackets like [^1][^2] — never combine numbers in one bracket
   (never write [^1, 2] or [^1,^2]).
3. If a claim cannot be supported by any provided source, explicitly state "insufficient evidence" instead of guessing.
4. Never fabricate information not present in the sources.
5. Synthesize across multiple sources when they provide complementary information. If
   several sources define the same term, write ONE clear definition — never repeat the
   same definition or claim more than once in different words. Never restate the same
   fact under two different headings.
6. Internal Document sources are authoritative and describe the user's own subject matter.
   When both kinds of source are present, ground the answer in the documents and use web
   results only to add context around them.
7. If an Internal Document identifies the person, company, or product being asked about,
   that is the subject of the question. Never describe a DIFFERENT entity that merely
   shares a name because a web source mentions it, and never present a list of same-named
   candidates as the answer. If a web source is clearly about someone or something else,
   ignore it entirely rather than citing it.
8. Write in plain text and standard markdown only. Never use LaTeX or math notation
   (no $...$, no \\(...\\), no \\frac, no subscript syntax) — write "d_model" or
   "the model dimension" instead of "$d_{model}$", and spell out formulas in plain words
   or simple inline notation (e.g. "Q times K-transpose, divided by the square root of d_k").
9. Structure your answer with clear paragraphs and bullet points where helpful.
10. Respond with ONLY the answer text itself — no JSON, no preamble, no code fences. Just the
   markdown answer with [^N] citation markers inline.
"""

# Appended only when the web was searched because the corpus returned nothing. Answering a
# question about the user's own documents from public web pages, without saying so, is what
# made the original failure so damaging — the answer looked identical to a real one.
WEB_FALLBACK_NOTICE = """

IMPORTANT — SOURCE DISCLOSURE: The user's uploaded documents contained nothing relevant to
this question, so every source below comes from a public web search. Open your answer with
one short sentence saying you could not find this in their documents and that the following
comes from the web. Then answer from the web sources. If those sources appear to be about a
different person, company, or product than the user meant, say that plainly instead of
presenting them as the answer."""

# Appended instead of WEB_FALLBACK_NOTICE when the web was searched because document
# retrieval itself FAILED (a search backend outage), not because the search ran and found
# nothing. Confirmed live: without this distinction, a Pinecone outage produced the answer
# "I could not find this in your documents" for a fact the documents definitely contain —
# false, and actively misleading about whether the user needs to upload anything.
WEB_FALLBACK_ERROR_NOTICE = """

IMPORTANT — SOURCE DISCLOSURE: The user's document search is temporarily unavailable, so
their documents could not actually be checked for this question — this is NOT the same as
searching them and finding nothing. Every source below comes from a public web search
instead. Open your answer with one short sentence saying document search is temporarily
unavailable and this answer comes from the web instead. Never say the documents "don't
contain" or "don't cover" this — that claim would be false, since they were never searched.
Then answer from the web sources, with the same rule against presenting a different person,
company, or product as if it were the one the user meant."""


async def synthesizer_node(state: AgentState) -> AgentState:
    """Generate a draft answer with inline [^N] citations from fused multi-source context."""
    query = state.get("query", "")
    fused_evidence: list[FusedEvidence] = state.get("fused_evidence", [])
    route_history = list(state.get("route_history", []))
    route_history.append("synthesizer")

    # If fused_evidence is not populated, construct it on-the-fly from chunks/web_results
    if not fused_evidence:
        chunks = state.get("retrieved_chunks", [])
        web_results = state.get("web_results", [])
        idx = 1
        for c in chunks:
            doc_id = c.metadata.get("document_id")
            fused_evidence.append(
                {
                    "source_id": idx,
                    "marker": f"[^{idx}]",
                    "source_type": "document",
                    "title": f"Document (Doc ID: {doc_id})",
                    "content": c.text,
                    "chunk_id": c.id,
                    "doc_id": doc_id,
                    "page": c.metadata.get("page"),
                    "section": c.metadata.get("section"),
                }
            )
            idx += 1
        for w in web_results:
            fused_evidence.append(
                {
                    "source_id": idx,
                    "marker": f"[^{idx}]",
                    "source_type": "web",
                    "title": w.get("title", "Web Source"),
                    "content": w.get("snippet", ""),
                    "url": w.get("url"),
                }
            )
            idx += 1

    if not fused_evidence:
        return {
            **state,
            "draft_answer": "I could not find sufficient information to answer this question. No relevant documents or web results were available.",
            "citations": [],
            "route_history": route_history,
        }

    # Format numbered source blocks for the prompt
    context_blocks: list[str] = []
    for ev in fused_evidence[:10]:
        stype = "Internal Document" if ev.get("source_type") == "document" else "Live Web Result"
        meta_str = f"Title: {ev.get('title', '')}"
        if ev.get("url"):
            meta_str += f"\nURL: {ev.get('url')}"
        if ev.get("section"):
            meta_str += f"\nSection: {ev.get('section')}"
        if ev.get("page"):
            meta_str += f"\nPage: {ev.get('page')}"

        context_blocks.append(
            f"[Source {ev['source_id']}] ({stype})\n"
            f"{meta_str}\n"
            f"Content: {ev.get('content', '')}"
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    # Format multi-turn conversation context for dialogue continuity.
    # chat_history is already the bounded sliding window (agents/memory/service.py
    # loads exactly MEMORY_WINDOW_MESSAGES) — no re-slicing needed here.
    chat_history = state.get("chat_history") or []
    conversation_summary = state.get("conversation_summary")
    episodic_memories = state.get("episodic_memories") or []
    history_blocks: list[str] = []

    if conversation_summary:
        history_blocks.append(f"[Prior Conversation Summary]: {conversation_summary}")

    for turn in chat_history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = turn.get("content", "").strip()
        if content:
            history_blocks.append(f"{role}: {content[:350]}")

    history_context = ""
    if history_blocks:
        history_context = "Conversation History (for context & conversational continuity):\n" + "\n".join(history_blocks) + "\n\n"

    episodic_context = ""
    if episodic_memories:
        episodic_lines = [
            f"- ({m.get('role', 'user')}, from another conversation): {m.get('text', '')[:300]}"
            for m in episodic_memories
        ]
        episodic_context = (
            "Relevant memory from earlier conversations (background only — never cite this as evidence):\n"
            + "\n".join(episodic_lines) + "\n\n"
        )

    raw_query = state.get("raw_query")
    query_context = f"User Question: {query}"
    if raw_query and raw_query != query:
        query_context += f" (Contextualized from original question: '{raw_query}')"

    synthesis_prompt = f"""{history_context}{episodic_context}Current {query_context}

Available Sources:
{context_text}

Generate a comprehensive answer with [^N] citation markers for each factual claim."""

    system_prompt = SYNTHESIS_SYSTEM_PROMPT
    if state.get("web_is_fallback"):
        system_prompt += WEB_FALLBACK_ERROR_NOTICE if state.get("retrieval_errored") else WEB_FALLBACK_NOTICE

    try:
        token_callback = state.get("token_callback")

        if token_callback is not None:
            # Streaming path (/chat/stream): tokens are pushed to the client live as
            # they arrive, while we also accumulate the full text below for citation
            # parsing and for the downstream critic/query_rewriter retry loop, which
            # both need the complete answer, not a stream.
            from agents.llm_helper import generate_gemini_content_stream

            text_pieces: list[str] = []
            async for piece in generate_gemini_content_stream(
                contents=synthesis_prompt,
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=1536,
                timeout=25.0,
            ):
                text_pieces.append(piece)
                await token_callback(piece)
            answer = "".join(text_pieces).strip()
        else:
            # Non-streaming path (/chat, and voice's deep_research): one blocking
            # call, full text back at once.
            from agents.llm_helper import generate_gemini_content

            gen_kwargs: dict = {
                "contents": synthesis_prompt,
                "system_instruction": system_prompt,
                "temperature": 0.2,
                "max_output_tokens": 1536,
                "timeout": 10.0,
            }

            if state.get("voice_fast_mode"):
                # Voice is a live phone call: the user is on the line for every
                # second, and the Live model compresses whatever we return into
                # 2-3 spoken sentences anyway, so a long draft is wasted work.
                # Cap the draft short and the budget tight.
                #
                # Deliberately does NOT pin a model list. An earlier version of
                # this block hard-coded gemini-3.5-flash-lite first on the
                # strength of one benchmark; re-measuring the next day showed
                # that same model at 10.9-17.0s while the models it demoted
                # answered in under 2s. Model speed here swings by ~10x within
                # the hour, so llm_helper measures it per call and orders the
                # chain from data — see _model_latency_ewma there.
                gen_kwargs["max_output_tokens"] = 700
                gen_kwargs["timeout"] = 8.0

            answer = (await generate_gemini_content(**gen_kwargs)).strip()

        # Citations used = whichever [^N] markers actually appear in the finished
        # answer, matched back against the numbered source list — never a separate
        # LLM-reported field, so the two can't drift out of sync with each other.
        # Each matched bracket may itself contain more than one number if the model
        # didn't follow the one-source-per-bracket rule, so extract every digit run
        # inside each bracket rather than assuming exactly one.
        cited_source_ids = {
            int(n)
            for bracket in CITATION_MARKER_RE.findall(answer)
            for n in CITATION_NUMBER_RE.findall(bracket)
        }
        citations: list[Citation] = []
        for ev in fused_evidence:
            if ev["source_id"] in cited_source_ids:
                citation: Citation = {
                    "marker": f"[^{ev['source_id']}]",
                    # `.get(key, default)` only falls back when the key is MISSING —
                    # web and graph evidence set "chunk_id" explicitly to None (they
                    # have no chunk), so `.get("chunk_id", "")` still returned None
                    # and 500'd every response that cited a web or graph source
                    # against Citation's non-nullable `chunk_id: str`.
                    "chunk_id": ev.get("chunk_id") or "",
                    "text_snippet": (ev.get("content") or "")[:200],
                    "source_type": ev.get("source_type", "document"),
                    "url": ev.get("url"),
                    "page": ev.get("page"),
                    "section": ev.get("section"),
                }
                citations.append(citation)

        logger.info(
            "Draft answer synthesized",
            answer_length=len(answer),
            citations_count=len(citations),
            sources_used=sorted(cited_source_ids),
        )

        return {
            **state,
            "draft_answer": answer,
            "citations": citations,
            "route_history": route_history,
        }

    except Exception as e:
        logger.error("Synthesizer LLM call failed", error=str(e))

        # This used to return the top evidence chunk verbatim as the "answer". That
        # reads to the user as a confident reply — it is prose from their own document,
        # with no citations and no verification — while actually meaning the model never
        # ran. Worse, the raw chunk usually contains the fact being asked about, so the
        # failure is invisible to anyone spot-checking the output. Say plainly that the
        # answer could not be generated, and offer the source we would have used.
        excerpt = ""
        if fused_evidence:
            top = fused_evidence[0]
            where = top.get("section") or top.get("title") or "the source document"
            excerpt = (
                f"\n\nThe most relevant passage found was in **{where}**, "
                f"which you can read directly:\n\n> {(top.get('content') or '')[:400].strip()}"
            )

        return {
            **state,
            "draft_answer": (
                "I could not generate an answer just now — the language model was "
                "unavailable, so nothing here has been written or verified by it."
                + excerpt
            ),
            "citations": [],
            "synthesis_failed": True,
            "route_history": route_history,
        }
