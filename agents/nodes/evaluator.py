"""Retrieval evaluator node — grades whether retrieved chunks are sufficient.

Uses Gemini 3.5 Flash-Lite to assess if the retrieved context can fully
answer the user's query. Returns a confidence score (0.0–1.0) and decides
whether web augmentation is needed.

Threshold: ≥ 0.7 → proceed to synthesis; < 0.7 → trigger web search.
"""

from __future__ import annotations

import json

import structlog

from agents.state import AgentState

logger = structlog.get_logger(__name__)

RELEVANCE_THRESHOLD = 0.7

EVALUATOR_SYSTEM_PROMPT = """You are a retrieval quality evaluator for a RAG system.

Given a user query and a set of retrieved document chunks, assess whether the chunks
contain sufficient information to fully and accurately answer the query.

Respond with ONLY valid JSON (no markdown, no extra text):
{
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<1-2 sentence explanation>",
  "missing_aspects": ["<aspect not covered>", ...]
}

Scoring guide:
- 1.0: Chunks directly and completely answer the query with specific evidence
- 0.7-0.9: Chunks cover most aspects but may lack some details
- 0.4-0.6: Chunks are partially relevant but significant gaps exist
- 0.0-0.3: Chunks are mostly irrelevant or query requires external/live information
"""


async def evaluator_node(state: AgentState) -> AgentState:
    """Grade retrieval quality and decide if web augmentation is needed."""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    route_history = list(state.get("route_history", []))
    route_history.append("evaluator")

    # No chunks at all → definitely need web search
    if not chunks:
        logger.info("No chunks retrieved, routing to web search")
        return {
            **state,
            "retrieval_confidence": 0.0,
            "needs_web_augmentation": True,
            "route_history": route_history,
        }

    # Build context from retrieved chunks
    context_text = "\n\n".join(
        f"[Chunk {i + 1}] (score: {c.rerank_score or c.dense_score or 0.0:.4f})\n{c.text}"
        for i, c in enumerate(chunks[:5])
    )

    eval_prompt = f"""User Query: {query}

Retrieved Context:
{context_text}

Evaluate whether this context is sufficient to answer the query completely."""

    try:
        from google import genai

        from api.config import get_settings

        settings = get_settings()
        client = genai.Client(api_key=settings.gemini_api_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=eval_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=EVALUATOR_SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )

        raw_text = response.text.strip()
        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(raw_text)
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")
        missing = result.get("missing_aspects", [])

        needs_web = confidence < RELEVANCE_THRESHOLD

        logger.info(
            "Retrieval evaluation complete",
            confidence=round(confidence, 3),
            needs_web_augmentation=needs_web,
            reasoning=reasoning,
            missing_aspects=missing,
        )

        return {
            **state,
            "retrieval_confidence": confidence,
            "needs_web_augmentation": needs_web,
            "route_history": route_history,
        }

    except Exception as e:
        logger.warning("Evaluator LLM call failed, using heuristic", error=str(e))

        # Heuristic fallback: use rerank scores
        top_score = max(
            (c.rerank_score or c.dense_score or 0.0 for c in chunks),
            default=0.0,
        )
        # Map rerank score to confidence (rerank scores typically 0–1)
        heuristic_confidence = min(top_score, 1.0)
        needs_web = heuristic_confidence < RELEVANCE_THRESHOLD

        return {
            **state,
            "retrieval_confidence": round(heuristic_confidence, 3),
            "needs_web_augmentation": needs_web,
            "route_history": route_history,
        }
