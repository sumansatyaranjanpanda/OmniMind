"""Query Rewriter node — autonomous re-search strategy for self-correction.

When the Citation Critic rejects a draft answer due to unverified claims or low
faithfulness, this node analyzes the critic's specific feedback and unsupported
claims to generate targeted, search-optimized sub-queries targeting the exact
missing evidence.
"""

from __future__ import annotations

import json

import structlog

from agents.state import AgentState, SubQuery

logger = structlog.get_logger(__name__)

REWRITER_SYSTEM_PROMPT = """You are an expert Query Reformulation Agent for an enterprise RAG system.

The previous draft answer failed verification because some claims lacked source evidence.
Your job: Analyze the unsupported claims and critic feedback to generate targeted, high-precision
sub-queries to retrieve the missing information.

Rules:
1. Generate 1-2 focused sub-queries directly targeting the specific unverified claims.
2. Default to source="internal". The user's own uploaded documents are the subject of
   almost every question here, and a first attempt that missed usually needs different
   wording against the same documents — not a different corpus.
3. Only assign source="web" when the missing information is inherently external and
   current (live pricing, today's news, public competitor data). Never switch to the web
   merely because internal retrieval came back thin: answering a question about the user's
   documents with public web pages about a similarly-named subject is worse than saying
   the documents do not cover it.
4. Make search queries concise and keyword-dense (no filler words). For internal
   sub-queries, try the user's own terminology rather than adding new qualifiers.

Output ONLY valid JSON matching this schema:
{
  "sub_queries": [
    {
      "query": "<targeted search string>",
      "source": "internal" | "web",
      "reasoning": "<why this query will find the missing evidence>"
    }
  ]
}
"""


async def query_rewriter_node(state: AgentState) -> AgentState:
    """Generate targeted re-search sub-queries based on critic feedback."""
    query = state.get("query", "")
    critic_feedback = state.get("critic_feedback", "")
    unsupported_claims = state.get("unsupported_claims", [])
    route_history = list(state.get("route_history", []))
    route_history.append("query_rewriter")

    logger.info(
        "Query Rewriter triggered",
        unsupported_claims_count=len(unsupported_claims),
        feedback=critic_feedback[:100],
    )

    try:
        from google import genai

        from api.config import get_settings

        settings = get_settings()
        client = genai.Client(api_key=settings.gemini_api_key)

        prompt = f"""Original Query: {query}
Critic Feedback: {critic_feedback}
Unsupported Claims: {json.dumps(unsupported_claims)}

Generate targeted sub-queries to find the missing evidence:"""

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=REWRITER_SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(raw_text)
        raw_subs = result.get("sub_queries", [])

        new_sub_queries: list[SubQuery] = [
            {
                "query": sq.get("query", query),
                # Defaulting to "web" here sent every unlabelled retry to the public
                # internet. Combined with the critic failing empty-evidence answers, that
                # turned "the documents didn't cover this" into "here is a stranger with
                # the same name" — retry against the user's own corpus instead.
                "source": sq.get("source", "internal"),
                "reasoning": sq.get("reasoning", "Corrective retry"),
            }
            for sq in raw_subs
            if isinstance(sq, dict) and sq.get("query")
        ]

        if not new_sub_queries:
            fallback_text = " ".join(unsupported_claims[:2]) if unsupported_claims else query
            new_sub_queries = [
                {"query": fallback_text, "source": "internal", "reasoning": "Fallback retry query"}
            ]

        logger.info(
            "Targeted retry sub-queries generated",
            sub_queries=[f"[{sq['source']}] {sq['query']}" for sq in new_sub_queries],
        )

        return {
            **state,
            "sub_queries": new_sub_queries,
            "route_history": route_history,
        }

    except Exception as e:
        logger.warning("Query Rewriter LLM call failed, using fallback query", error=str(e))
        return {
            **state,
            "sub_queries": [
                {"query": query, "source": "internal", "reasoning": "Fallback on rewriter exception"}
            ],
            "route_history": route_history,
        }
