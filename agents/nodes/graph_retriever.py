"""Graph Retriever node — traverses tenant knowledge graph for multi-hop entity relationships.

Extracts key entities from the user query and traverses the KnowledgeGraphStore to
fetch connected entity subgraphs, relationship paths, and supporting evidence.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from agents.state import AgentState
from retrieval.graph_store import KnowledgeGraphStore

logger = structlog.get_logger(__name__)


async def graph_retriever_node(state: AgentState) -> AgentState:
    """Traverse the knowledge graph to find multi-hop relationships around query entities."""
    query = state.get("query", "")
    tenant_id = state.get("tenant_id", "default")
    route_history = list(state.get("route_history", []))
    route_history.append("graph_retriever")

    logger.info("Graph Retriever node executing", query=query, tenant_id=tenant_id)

    # 1. Identify target entities to search in the graph
    sub_queries = state.get("sub_queries", [])
    graph_queries = [sq["query"] for sq in sub_queries if sq.get("source") in ("graph", "internal")]
    if not graph_queries:
        graph_queries = [query]

    discovered_paths: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for q in graph_queries:
        # Search for matching entities in the tenant's graph
        matched_entities = KnowledgeGraphStore.search_entities(tenant_id, q, top_k=3)

        # Also extract capitalized words / noun phrases from query as potential entities
        words = re.findall(r"\b[A-Z][a-zA-Z0-9_-]+\b", q)
        for word in words:
            matched_entities.extend(KnowledgeGraphStore.search_entities(tenant_id, word, top_k=2))

        # Deduplicate matched entities by name
        unique_entities = {e["name"]: e for e in matched_entities}.values()

        for entity in unique_entities:
            neighborhood = KnowledgeGraphStore.get_neighborhood(
                tenant_id=tenant_id,
                entity_name=entity["name"],
                max_hops=2,
            )

            for item in neighborhood:
                edge_key = (item["source"], item["relation"], item["target"])
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    discovered_paths.append(item)

    logger.info(
        "Graph Retriever completed",
        paths_found=len(discovered_paths),
        entities_searched=len(graph_queries),
    )

    return {
        **state,
        "graph_paths": discovered_paths,
        "route_history": route_history,
    }
