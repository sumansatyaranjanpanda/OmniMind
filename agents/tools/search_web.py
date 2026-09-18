"""Web search tool — Multi-provider external search for knowledge gaps.

When the retrieval evaluator determines that internal documents are
insufficient (confidence < 0.7), this tool fetches live web results
to augment the context.

Providers:
1. Tavily AI Search (Primary, RAG-native, deep markdown extracts) — requires TAVILY_API_KEY.
2. DuckDuckGo Search (Zero-config fallback, 100% free, no key required).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from api.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

TAVILY_API_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5
TIMEOUT_SECONDS = 10


async def search_web(query: str, max_results: int = MAX_RESULTS) -> list[dict[str, Any]]:
    """Search the web using Tavily AI (with DuckDuckGo fallback) and return structured results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    # 1. Primary: Tavily AI Search (if API key is present)
    if settings.tavily_api_key:
        try:
            results = await _tavily_search(clean_query, max_results)
            if results:
                logger.info(
                    "Tavily AI search succeeded",
                    provider="tavily",
                    query=clean_query,
                    result_count=len(results),
                )
                return results
        except Exception as e:
            logger.warning(
                "Tavily search failed; falling back to DuckDuckGo",
                error=str(e),
                query=clean_query,
            )

    # 2. Fallback: DuckDuckGo Search (zero-config, free)
    try:
        results = await _ddgs_search(clean_query, max_results)
        if results:
            logger.info(
                "DuckDuckGo search succeeded",
                provider="duckduckgo",
                query=clean_query,
                result_count=len(results),
            )
            return results
    except Exception as e:
        logger.warning("DuckDuckGo search failed, trying HTTP fallback", error=str(e))

    # 3. Last-resort minimal HTTP fallback
    return await _http_fallback_search(clean_query, max_results)


async def _tavily_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Execute search query using Tavily AI's RAG-optimized REST API."""
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
        "max_results": max_results,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(TAVILY_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", "Web Result"),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            })

        return results[:max_results]


async def _ddgs_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Search using the ddgs / duckduckgo-search Python library."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    def _search_sync():
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
                for r in results
            ]

    return await asyncio.to_thread(_search_sync)


async def _http_fallback_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Minimal HTTP-based DuckDuckGo instant answer fallback."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            data = response.json()

            results = []
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", "DuckDuckGo Result"),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("AbstractText", ""),
                })

            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                    })

            return results[:max_results]

    except Exception as e:
        logger.warning("HTTP fallback search failed", error=str(e))
        return []
