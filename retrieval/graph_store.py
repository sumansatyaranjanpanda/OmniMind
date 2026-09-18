"""Multi-Tenant Knowledge Graph Store using NetworkX with multi-hop traversal.

Features:
1. Multi-Tenant Isolated DiGraphs: Each tenant has a separate graph namespace.
2. Neighborhood Extraction (Local Search): Retrieves 1-to-2 hop subgraphs around query entities.
3. Multi-Hop Path Finding: Discovers multi-step relationships connecting disparate entities.
4. Community Detection (Global Search): Detects connected entity clusters for high-level summaries.
5. JSON Export/Import for persistence.
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
import structlog

from ingestion.graph_extractor import Entity, ExtractedGraph, Relationship

logger = structlog.get_logger(__name__)

# Global in-memory multi-tenant graph storage: {tenant_id: nx.DiGraph}
_tenant_graphs: dict[str, nx.DiGraph] = {}


def get_tenant_graph(tenant_id: str) -> nx.DiGraph:
    """Get or create the NetworkX directed graph for a given tenant."""
    if tenant_id not in _tenant_graphs:
        _tenant_graphs[tenant_id] = nx.DiGraph()
    return _tenant_graphs[tenant_id]


class KnowledgeGraphStore:
    """Operations interface for querying and modifying tenant knowledge graphs."""

    @staticmethod
    def add_extracted_graph(tenant_id: str, graph_data: ExtractedGraph) -> dict[str, int]:
        """Ingest extracted entities and relationships into the tenant's graph."""
        graph = get_tenant_graph(tenant_id)
        nodes_added = 0
        edges_added = 0

        # Add / update entities
        for entity in graph_data.entities:
            norm_name = entity.name.strip()
            if not norm_name:
                continue

            if norm_name not in graph:
                graph.add_node(
                    norm_name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    doc_ids=set(),
                    chunk_ids=set(),
                )
                nodes_added += 1
            else:
                # Merge description if empty
                if not graph.nodes[norm_name].get("description") and entity.description:
                    graph.nodes[norm_name]["description"] = entity.description

        # Add / update relationships
        for rel in graph_data.relationships:
            src = rel.source.strip()
            tgt = rel.target.strip()
            if not src or not tgt or src == tgt:
                continue

            # Ensure endpoints exist
            if src not in graph:
                graph.add_node(src, entity_type="CONCEPT", description="", doc_ids=set(), chunk_ids=set())
                nodes_added += 1
            if tgt not in graph:
                graph.add_node(tgt, entity_type="CONCEPT", description="", doc_ids=set(), chunk_ids=set())
                nodes_added += 1

            # Update doc/chunk references
            if rel.doc_id:
                graph.nodes[src]["doc_ids"].add(rel.doc_id)
                graph.nodes[tgt]["doc_ids"].add(rel.doc_id)
            if rel.chunk_id:
                graph.nodes[src]["chunk_ids"].add(rel.chunk_id)
                graph.nodes[tgt]["chunk_ids"].add(rel.chunk_id)

            # Add directed edge with relation attributes
            graph.add_edge(
                src,
                tgt,
                relation=rel.relation,
                confidence=rel.confidence,
                evidence=rel.evidence_snippet,
                doc_id=rel.doc_id,
                chunk_id=rel.chunk_id,
            )
            edges_added += 1

        logger.info(
            "Graph ingestion complete",
            tenant_id=tenant_id,
            nodes_added=nodes_added,
            edges_added=edges_added,
            total_nodes=graph.number_of_nodes(),
            total_edges=graph.number_of_edges(),
        )

        return {"nodes_added": nodes_added, "edges_added": edges_added}

    @staticmethod
    def search_entities(tenant_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find matching entities in the tenant graph via keyword/substring match."""
        graph = get_tenant_graph(tenant_id)
        q_lower = query.lower().strip()
        matches = []

        for node, attrs in graph.nodes(data=True):
            node_lower = node.lower()
            score = 0.0
            if q_lower == node_lower:
                score = 1.0
            elif q_lower in node_lower or node_lower in q_lower:
                score = 0.8
            else:
                # Check description
                desc = attrs.get("description", "").lower()
                if q_lower in desc:
                    score = 0.5

            if score > 0.0:
                matches.append({
                    "name": node,
                    "entity_type": attrs.get("entity_type", "CONCEPT"),
                    "description": attrs.get("description", ""),
                    "score": score,
                    "degree": graph.degree(node),
                })

        # Sort by match score then degree (connectedness)
        matches.sort(key=lambda x: (x["score"], x["degree"]), reverse=True)
        return matches[:top_k]

    @staticmethod
    def get_neighborhood(
        tenant_id: str,
        entity_name: str,
        max_hops: int = 2,
    ) -> list[dict[str, Any]]:
        """Extract all 1-to-N hop relationships and connected entities around a target entity."""
        graph = get_tenant_graph(tenant_id)
        if entity_name not in graph:
            # Try finding closest match
            matches = KnowledgeGraphStore.search_entities(tenant_id, entity_name, top_k=1)
            if not matches:
                return []
            entity_name = matches[0]["name"]

        # Extract ego graph (subgraph of radius max_hops)
        subgraph = nx.ego_graph(graph.to_undirected(), entity_name, radius=max_hops)
        results = []

        # Find directed edges in the subgraph
        for u, v in subgraph.edges():
            # Check edge in forward direction
            if graph.has_edge(u, v):
                edge_data = graph.get_edge_data(u, v, default={})
                results.append({
                    "source": u,
                    "target": v,
                    "relation": edge_data.get("relation", "relates_to"),
                    "confidence": edge_data.get("confidence", 1.0),
                    "evidence": edge_data.get("evidence", ""),
                    "doc_id": edge_data.get("doc_id"),
                    "chunk_id": edge_data.get("chunk_id"),
                    "path_str": f"({u}) ──[{edge_data.get('relation', 'relates_to')}]──► ({v})",
                })
            # Check edge in reverse direction
            if graph.has_edge(v, u):
                edge_data = graph.get_edge_data(v, u, default={})
                results.append({
                    "source": v,
                    "target": u,
                    "relation": edge_data.get("relation", "relates_to"),
                    "confidence": edge_data.get("confidence", 1.0),
                    "evidence": edge_data.get("evidence", ""),
                    "doc_id": edge_data.get("doc_id"),
                    "chunk_id": edge_data.get("chunk_id"),
                    "path_str": f"({v}) ──[{edge_data.get('relation', 'relates_to')}]──► ({u})",
                })

        return results

    @staticmethod
    def find_paths(
        tenant_id: str,
        source_entity: str,
        target_entity: str,
        max_depth: int = 3,
    ) -> list[list[str]]:
        """Find all multi-hop paths connecting two entities across the knowledge graph."""
        graph = get_tenant_graph(tenant_id)
        if source_entity not in graph or target_entity not in graph:
            return []

        try:
            paths = list(nx.all_simple_paths(graph, source_entity, target_entity, cutoff=max_depth))
            return paths
        except nx.NetworkXNoPath:
            return []

    @staticmethod
    def get_community_summary(tenant_id: str) -> list[dict[str, Any]]:
        """Perform community detection (Louvain/greedy modularity) to extract global themes."""
        graph = get_tenant_graph(tenant_id)
        if graph.number_of_nodes() < 2:
            return []

        undirected = graph.to_undirected()
        try:
            communities = list(nx.community.greedy_modularity_communities(undirected))
            summaries = []
            for i, comm in enumerate(communities[:5]):
                members = list(comm)
                # Find most central node in this community
                sub = undirected.subgraph(members)
                centralities = nx.degree_centrality(sub)
                top_nodes = sorted(centralities.keys(), key=lambda n: centralities[n], reverse=True)[:5]

                summaries.append({
                    "community_id": i + 1,
                    "size": len(members),
                    "core_entities": top_nodes,
                    "all_entities": members,
                })
            return summaries
        except Exception as e:
            logger.debug("Community detection skipped", error=str(e))
            return []

    @staticmethod
    def export_to_json(tenant_id: str) -> str:
        """Export the graph as JSON for serialization and API visualization."""
        graph = get_tenant_graph(tenant_id)
        nodes = [{"id": n, **{k: list(v) if isinstance(v, set) else v for k, v in attrs.items()}} for n, attrs in graph.nodes(data=True)]
        edges = [{"source": u, "target": v, **attrs} for u, v, attrs in graph.edges(data=True)]
        return json.dumps({"tenant_id": tenant_id, "nodes": nodes, "edges": edges})

    @staticmethod
    def load_from_json(tenant_id: str, json_str: str) -> None:
        """Import a graph from JSON representation."""
        data = json.loads(json_str)
        graph = get_tenant_graph(tenant_id)
        graph.clear()

        for node in data.get("nodes", []):
            node_id = node.pop("id")
            if "doc_ids" in node:
                node["doc_ids"] = set(node["doc_ids"])
            if "chunk_ids" in node:
                node["chunk_ids"] = set(node["chunk_ids"])
            graph.add_node(node_id, **node)

        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            graph.add_edge(src, tgt, **edge)
