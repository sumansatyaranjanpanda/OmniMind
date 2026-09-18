"""Query rewriter and expansion module for hybrid retrieval.

Analyzes raw user queries, decomposes complex multi-intent questions into sub-queries,
and extracts key technical terms/keywords to maximize both dense semantic and sparse
lexical (BM25) search recall.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx
import structlog

from api.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


@dataclass
class QueryExpansionResult:
    """Represents the expanded queries and keywords for multi-stage search."""

    original_query: str
    rewritten_query: str
    sub_queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def all_search_queries(self) -> list[str]:
        """Returns unique list of queries to execute across retrievers."""
        queries = [self.original_query]
        if self.rewritten_query and self.rewritten_query != self.original_query:
            queries.append(self.rewritten_query)
        for sq in self.sub_queries:
            if sq and sq not in queries:
                queries.append(sq)
        return queries


QUERY_REWRITE_SYSTEM_PROMPT = """You are a search query optimizer for an enterprise RAG system.
Given a user's raw search query, generate an optimized search strategy in JSON format with:
1. "rewritten_query": A clean, fully specified version of the query with pronouns and ambiguity resolved.
2. "sub_queries": 1 to 3 distinct sub-questions exploring different angles or specific aspects of the query.
3. "keywords": 3 to 6 key technical terms, acronyms, table names, or specific entities for lexical/BM25 matching.

Return ONLY valid JSON matching this schema:
{
  "rewritten_query": "...",
  "sub_queries": ["...", "..."],
  "keywords": ["...", "..."]
}
"""


async def rewrite_query(query: str) -> QueryExpansionResult:
    """Rewrite and expand a user query using Gemini Flash-Lite.

    Falls back cleanly to basic keyword extraction if the LLM is unavailable or times out.
    """
    clean_query = query.strip()
    if not clean_query:
        return QueryExpansionResult(
            original_query=query,
            rewritten_query=query,
            sub_queries=[],
            keywords=[],
        )

    # If no API key configured, use local rule-based fallback
    api_key = settings.gemini_api_key
    if not api_key:
        logger.debug("No GEMINI_API_KEY configured; using fallback query expansion")
        words = [w for w in re.findall(r"\b\w+\b", clean_query) if len(w) > 2]
        return QueryExpansionResult(
            original_query=clean_query,
            rewritten_query=clean_query,
            sub_queries=[],
            keywords=words,
        )

    try:
        from agents.llm_helper import generate_gemini_content

        contents = f"User Query: {clean_query}"
        raw_text = await generate_gemini_content(
            contents=contents,
            system_instruction=QUERY_REWRITE_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=300,
            response_mime_type="application/json",
            # Query expansion, not answer quality — lite-first cuts latency/cost, and
            # this runs on every retrieval call including retries.
            candidate_models=["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash"],
        )

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw_text)

        rewritten = parsed.get("rewritten_query", clean_query)
        sub_queries = parsed.get("sub_queries", [])
        keywords = parsed.get("keywords", [])

        logger.info(
            "Query rewritten successfully",
            original=clean_query,
            rewritten=rewritten,
            sub_query_count=len(sub_queries),
            keyword_count=len(keywords),
        )

        return QueryExpansionResult(
            original_query=clean_query,
            rewritten_query=rewritten,
            sub_queries=sub_queries,
            keywords=keywords,
        )

    except Exception as e:
        logger.warning(
            "LLM query rewriting failed; falling back to heuristic expansion",
            error=str(e),
            query=clean_query,
        )
        words = [w for w in re.findall(r"\b\w+\b", clean_query) if len(w) > 2]
        return QueryExpansionResult(
            original_query=clean_query,
            rewritten_query=clean_query,
            sub_queries=[],
            keywords=words,
        )
