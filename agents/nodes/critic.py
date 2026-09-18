"""Citation Critic node — faithfulness verifier and hallucination blocker.

The critic is the final quality gate before any answer reaches the user.
It cross-checks every claim in the draft answer against the fused evidence
(document chunks and web results), calculating a faithfulness score (0.0–1.0).

- Score ≥ 0.85: VERIFIED → proceed to final output
- Score < 0.85 and retries remaining → route to query_rewriter for targeted re-search
- Max retries reached → finalize with appropriate transparency disclaimer
"""

from __future__ import annotations

import json
import re

import structlog

from agents.state import AgentState, FusedEvidence

logger = structlog.get_logger(__name__)

FAITHFULNESS_THRESHOLD = 0.85

_CITATION_RE = re.compile(r"\[\^[\d,\s\^]+\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Closing courtesies carry no claim, so they shouldn't force a verification round
# trip. Everything else that isn't a heading is treated as a claim — see
# _uncited_assertions for why the bias runs that way.
_NON_CLAIM_RE = re.compile(
    r"^(?:let me know|feel free|please (?:let me|reach|refer|verify|note)|"
    r"if you (?:need|have|would)|i (?:hope|can|could) )",
    re.IGNORECASE,
)


def _is_structural(line: str) -> bool:
    """Markdown scaffolding rather than a claim: headings and label-style lead-ins.

    Excluded explicitly rather than by length, because the synthesis prompt asks for
    headings and bullets — so a plain length cutoff would either treat every heading
    as an unverified claim (never taking the fast path) or let genuinely short claims
    through unchecked.
    """
    stripped = line.strip()
    return stripped.startswith("#") or stripped.endswith(":")


def _uncited_assertions(answer: str) -> list[str]:
    """Sentences that assert something checkable but carry no citation marker.

    This is the same question the LLM critic is asked, answered the cheap way for the
    case where the answer is unambiguously well-formed.

    Deliberately biased to over-report. An earlier version only counted a sentence as
    a claim if it contained a digit or a proper noun, which let an ordinary lowercase
    assertion ("managers approve requests in writing") pass unverified. Since the whole
    point of this gate is catching fabrication, any substantive prose without a marker
    now counts — the worst case is paying for a verification we didn't need, never
    skipping one we did.
    """
    uncited: list[str] = []
    for raw in _SENTENCE_RE.split(answer):
        if _is_structural(raw):
            continue
        sentence = raw.strip().lstrip("-*•# ").strip()
        if len(sentence) < 25:
            continue  # fragments and list labels carry nothing to hallucinate
        if _CITATION_RE.search(sentence):
            continue
        if _NON_CLAIM_RE.match(sentence):
            continue
        uncited.append(sentence)
    return uncited

CRITIC_SYSTEM_PROMPT = """You are a citation verification critic for an enterprise RAG system.

Your job: Verify that EVERY factual claim in the draft answer is strictly supported by
the provided source evidence. You are the last line of defense against hallucination.

For each claim in the answer:
1. Check if a citation marker [^N] is present.
2. Check if the cited source actually contains and supports the claim.
3. Identify any claims that lack citations or are unsupported.

Respond with ONLY valid JSON (no markdown fences):
{
  "faithfulness_score": <float 0.0 to 1.0>,
  "verification_status": "<VERIFIED|PARTIALLY_VERIFIED|INSUFFICIENT_EVIDENCE>",
  "feedback": "<specific 1-2 sentence feedback on what failed or passed>",
  "unsupported_claims": ["<specific factual claim that lacks source evidence>", ...],
  "verified_claims": ["<claim that is properly cited and backed>", ...]
}

Scoring guide:
- 1.0: Every claim is fully supported with a valid citation marker
- 0.85-0.99: Almost all claims are backed, only minor stylistic differences
- 0.5-0.84: Several key claims lack evidence or citations
- 0.0-0.49: Most claims are unsupported or fabricated
"""


async def critic_node(state: AgentState) -> AgentState:
    """Verify the draft answer's faithfulness against fused source evidence."""
    draft_answer = state.get("draft_answer", "")
    fused_evidence: list[FusedEvidence] = state.get("fused_evidence", [])
    citations = state.get("citations", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)
    route_history = list(state.get("route_history", []))
    route_history.append("critic")

    if not draft_answer:
        return {
            **state,
            "faithfulness_score": 0.0,
            "verification_status": "INSUFFICIENT_EVIDENCE",
            "critic_feedback": "No draft answer to verify.",
            "unsupported_claims": [],
            "iteration": iteration + 1,
            "route_history": route_history,
        }

    # The model was unreachable, so draft_answer is our apology rather than generated
    # text. There is nothing to verify, and retrying would re-run retrieval and
    # synthesis against a model that is still down — pure added latency on top of an
    # outage. Stop the correction loop here by exhausting the iteration budget.
    if state.get("synthesis_failed"):
        logger.info("Citation critic skipped: synthesis did not run (model unavailable)")
        return {
            **state,
            "faithfulness_score": 0.0,
            "verification_status": "INSUFFICIENT_EVIDENCE",
            "critic_feedback": "The language model was unavailable, so no answer was generated to verify.",
            "unsupported_claims": [],
            "final_answer": draft_answer,
            "iteration": max_iterations,
            "route_history": route_history,
        }

    # Nothing to verify against. Asking the model to check an answer for faithfulness
    # to an empty source list can only produce a guess, so spend nothing on it.
    if not fused_evidence and not state.get("retrieved_chunks") and not state.get("web_results"):
        logger.info("Citation critic skipped: no evidence was retrieved to verify against")
        return {
            **state,
            "faithfulness_score": 0.0,
            "verification_status": "INSUFFICIENT_EVIDENCE",
            "critic_feedback": "No source evidence was retrieved for this query.",
            "unsupported_claims": [],
            "final_answer": draft_answer if iteration + 1 >= max_iterations else "",
            "iteration": iteration + 1,
            "route_history": route_history,
        }

    # Fast path: if every assertive sentence already carries a citation marker, the
    # LLM critic has nothing to find — in testing it returns 1.0 on exactly this shape.
    # Verifying it costs a full model round trip on the *common* case, which is the
    # single largest avoidable chunk of latency on the RAG path. Skip only when the
    # answer is clean; anything doubtful still gets the real critic below.
    if citations:
        uncited = _uncited_assertions(draft_answer)
        if not uncited:
            logger.info(
                "Citation critic fast-path: every assertion is cited, skipping LLM verification",
                citations_count=len(citations),
                iteration=iteration,
            )
            return {
                **state,
                "faithfulness_score": 1.0,
                "verification_status": "VERIFIED",
                "critic_feedback": "All assertive sentences carry citation markers.",
                "unsupported_claims": [],
                "final_answer": draft_answer,
                "iteration": iteration + 1,
                "route_history": route_history,
            }
        logger.debug("Critic fast-path declined", uncited_count=len(uncited))

    # Build source text representation from fused evidence (or fallback to chunks/web)
    source_texts = []
    if fused_evidence:
        for ev in fused_evidence[:10]:
            source_texts.append(f"[Source {ev['source_id']}] {ev.get('content', '')}")
    else:
        for i, chunk in enumerate(state.get("retrieved_chunks", [])[:6]):
            source_texts.append(f"[Source {i + 1}] {chunk.text}")
        for j, web in enumerate(state.get("web_results", [])[:3]):
            source_texts.append(f"[Source {len(state.get('retrieved_chunks', [])) + j + 1}] {web.get('snippet', '')}")

    critic_prompt = f"""Draft Answer to Verify:
{draft_answer}

Source Evidence Available:
{chr(10).join(source_texts)}

Citations Provided in Draft: {json.dumps([c.get('marker', '') for c in citations])}

Verify every factual claim in the draft answer against the available sources."""

    try:
        from agents.llm_helper import generate_gemini_content

        raw_text = await generate_gemini_content(
            contents=critic_prompt,
            system_instruction=CRITIC_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=1024,
            response_mime_type="application/json",
            # Verification against provided evidence, not open-ended generation —
            # lite-first cuts latency/cost without a real quality tradeoff here.
            candidate_models=["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash"],
        )

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(raw_text)
        faithfulness = float(result.get("faithfulness_score", 0.5))
        status = result.get("verification_status", "PARTIALLY_VERIFIED")
        feedback = result.get("feedback", "")
        unsupported = result.get("unsupported_claims", [])

        logger.info(
            "Citation critic evaluation complete",
            faithfulness_score=round(faithfulness, 3),
            verification_status=status,
            unsupported_claims_count=len(unsupported),
            iteration=iteration,
            feedback=feedback[:100],
        )

        # Determine final answer and status
        if faithfulness >= FAITHFULNESS_THRESHOLD:
            final_answer = draft_answer
            status = "VERIFIED"
        elif iteration + 1 >= max_iterations:
            # Max retries reached — return best effort with transparency disclaimer
            if faithfulness >= 0.5:
                final_answer = draft_answer + "\n\n> ⚠️ *Some claims in this answer may have limited source support.*"
                status = "PARTIALLY_VERIFIED"
            else:
                final_answer = (
                    "I found some relevant information but could not fully verify "
                    "all claims against the source documents. Here is what I found:\n\n"
                    + draft_answer
                    + "\n\n> ⚠️ *Please verify these claims against the original documents.*"
                )
                status = "INSUFFICIENT_EVIDENCE"
        else:
            # Not verified and retries remain — will trigger Query Rewriter loop
            final_answer = ""

        return {
            **state,
            "faithfulness_score": round(faithfulness, 3),
            "verification_status": status,
            "critic_feedback": feedback,
            "unsupported_claims": unsupported,
            "final_answer": final_answer,
            "iteration": iteration + 1,
            "route_history": route_history,
        }

    except Exception as e:
        logger.warning("Critic LLM call failed, using heuristic", error=str(e))

        heuristic_score = 0.75 if citations else 0.4
        final_answer = draft_answer if heuristic_score >= 0.5 else ""

        return {
            **state,
            "faithfulness_score": heuristic_score,
            "verification_status": "PARTIALLY_VERIFIED" if heuristic_score >= 0.5 else "INSUFFICIENT_EVIDENCE",
            "critic_feedback": f"Critic LLM unavailable; heuristic score={heuristic_score}",
            "unsupported_claims": [],
            "final_answer": final_answer,
            "iteration": iteration + 1,
            "route_history": route_history,
        }
