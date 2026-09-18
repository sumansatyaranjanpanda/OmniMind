"""Web search augmentation node — executes targeted web searches for gap filling.

Queries external search providers (Tavily AI with DuckDuckGo fallback) specifically
for sub-queries assigned to web search by the Query Analyzer or Query Rewriter.
"""

from __future__ import annotations

import asyncio

import structlog

from agents.state import AgentState
from agents.tools.search_web import search_web

logger = structlog.get_logger(__name__)


async def web_search_node(state: AgentState) -> AgentState:
    """Fetch external web results for all planned web sub-queries."""
    route_history = list(state.get("route_history", []))
    route_history.append("web_search")

    sub_queries = state.get("sub_queries", [])
    web_queries = [sq["query"] for sq in sub_queries if sq.get("source") == "web"]

    # Fallback to main query if no web sub-queries were explicitly planned
    if not web_queries:
        web_queries = [state.get("query", "")]

    logger.info(
        "Web search node executing",
        web_queries=web_queries,
    )

    all_web_results: list[dict] = []
    seen_urls: set[str] = set()

    clean_queries = [q.strip() for q in web_queries if q.strip()]
    if not clean_queries:
        return {**state, "web_results": [], "route_history": route_history}

    try:
        # Sub-queries are independent searches against an external API, so issuing them
        # together turns N sequential network round-trips into one. return_exceptions
        # keeps a single failed search from discarding the results the others returned —
        # same pattern retriever_node already uses for its sub-queries.
        per_query = await asyncio.gather(
            *[search_web(q, max_results=4) for q in clean_queries],
            return_exceptions=True,
        )

        failures = 0
        for results in per_query:
            if isinstance(results, BaseException):
                failures += 1
                logger.warning("Web sub-query failed; continuing with the rest", error=str(results))
                continue
            for r in results:
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_web_results.append(r)
                elif not url:
                    all_web_results.append(r)

        logger.info(
            "Web search completed",
            total_results=len(all_web_results),
            failed_sub_queries=failures,
            sources=[r.get("url", "")[:60] for r in all_web_results[:3]],
        )

        return {
            **state,
            "web_results": all_web_results,
            "route_history": route_history,
        }

    except Exception as e:
        logger.warning("Web search node failed", error=str(e))
        return {
            **state,
            "web_results": [],
            "route_history": route_history,
        }
