"""Unit tests for Phase 6 Knowledge Graph and GraphRAG Engine."""

from __future__ import annotations

import pytest

from agents.graph import build_graph, route_after_analyzer, route_after_rewriter
from agents.nodes.graph_retriever import graph_retriever_node
from agents.state import AgentState
from ingestion.graph_extractor import Entity, ExtractedGraph, Relationship, extract_knowledge_triplets
from retrieval.graph_store import KnowledgeGraphStore, get_tenant_graph


# ── Entity & Extracted Graph Tests ──────────────────────────────


def test_entity_and_relationship_dataclass():
    """Entity and Relationship dataclasses initialize and serialize cleanly."""
    e = Entity(name="Transformer", entity_type="TECHNOLOGY", description="Attention-based neural architecture")
    r = Relationship(source="Transformer", target="Attention", relation="uses", confidence=0.95)

    graph = ExtractedGraph(entities=[e], relationships=[r])
    d = graph.to_dict()

    assert len(d["entities"]) == 1
    assert d["entities"][0]["name"] == "Transformer"
    assert len(d["relationships"]) == 1
    assert d["relationships"][0]["relation"] == "uses"


@pytest.mark.asyncio
async def test_heuristic_extraction_fallback():
    """Heuristic extraction finds key AI concepts when LLM is unavailable."""
    sample_text = "The Transformer architecture relies on Scaled Dot-Product Attention developed by Vaswani et al."
    graph = await extract_knowledge_triplets(sample_text, doc_id="doc_1", chunk_id="chk_1")

    assert len(graph.entities) >= 2
    entity_names = [e.name for e in graph.entities]
    assert "Transformer" in entity_names or "Attention" in entity_names


# ── KnowledgeGraphStore Multi-Tenant Tests ───────────────────────


def test_graph_store_ingestion_and_search():
    """KnowledgeGraphStore adds entities, creates edges, and enables search."""
    tenant = "test_graph_tenant_01"
    g = get_tenant_graph(tenant)
    g.clear()

    e1 = Entity(name="Vaswani et al.", entity_type="PERSON", description="Lead authors")
    e2 = Entity(name="Transformer", entity_type="TECHNOLOGY", description="Model architecture")
    e3 = Entity(name="Scaled Dot-Product Attention", entity_type="ALGORITHM", description="Core attention formula")

    r1 = Relationship(source="Vaswani et al.", target="Transformer", relation="authored", doc_id="doc_123")
    r2 = Relationship(source="Transformer", target="Scaled Dot-Product Attention", relation="contains", doc_id="doc_123")

    extracted = ExtractedGraph(entities=[e1, e2, e3], relationships=[r1, r2])
    res = KnowledgeGraphStore.add_extracted_graph(tenant, extracted)

    assert res["nodes_added"] == 3
    assert res["edges_added"] == 2

    # Search entities
    matches = KnowledgeGraphStore.search_entities(tenant, "transformer", top_k=2)
    assert len(matches) >= 1
    assert matches[0]["name"] == "Transformer"


def test_graph_store_neighborhood_traversal():
    """KnowledgeGraphStore extracts 1-to-2 hop subgraphs around an entity."""
    tenant = "test_graph_tenant_02"
    g = get_tenant_graph(tenant)
    g.clear()

    e1 = Entity(name="GPT-4", entity_type="TECHNOLOGY")
    e2 = Entity(name="OpenAI", entity_type="ORGANIZATION")
    e3 = Entity(name="Transformer", entity_type="CONCEPT")

    r1 = Relationship(source="OpenAI", target="GPT-4", relation="developed")
    r2 = Relationship(source="GPT-4", target="Transformer", relation="based_on")

    extracted = ExtractedGraph(entities=[e1, e2, e3], relationships=[r1, r2])
    KnowledgeGraphStore.add_extracted_graph(tenant, extracted)

    # Neighborhood around GPT-4
    neighborhood = KnowledgeGraphStore.get_neighborhood(tenant, "GPT-4", max_hops=1)
    assert len(neighborhood) == 2

    path_strings = [p["path_str"] for p in neighborhood]
    assert any("OpenAI" in s for s in path_strings)
    assert any("Transformer" in s for s in path_strings)


def test_graph_store_multi_hop_path_finding():
    """KnowledgeGraphStore discovers indirect paths connecting disparate entities."""
    tenant = "test_graph_tenant_03"
    g = get_tenant_graph(tenant)
    g.clear()

    e1 = Entity(name="Vaswani")
    e2 = Entity(name="Transformer")
    e3 = Entity(name="BERT")
    e4 = Entity(name="RoBERTa")

    r1 = Relationship(source="Vaswani", target="Transformer", relation="authored")
    r2 = Relationship(source="Transformer", target="BERT", relation="powers")
    r3 = Relationship(source="BERT", target="RoBERTa", relation="improved_by")

    extracted = ExtractedGraph(entities=[e1, e2, e3, e4], relationships=[r1, r2, r3])
    KnowledgeGraphStore.add_extracted_graph(tenant, extracted)

    # Find paths from Vaswani to RoBERTa (3 hops)
    paths = KnowledgeGraphStore.find_paths(tenant, "Vaswani", "RoBERTa", max_depth=3)
    assert len(paths) == 1
    assert paths[0] == ["Vaswani", "Transformer", "BERT", "RoBERTa"]


def test_graph_store_json_export_and_import():
    """KnowledgeGraphStore can serialize to JSON and restore perfectly."""
    tenant = "test_graph_tenant_04"
    g = get_tenant_graph(tenant)
    g.clear()

    extracted = ExtractedGraph(
        entities=[Entity(name="NodeA"), Entity(name="NodeB")],
        relationships=[Relationship(source="NodeA", target="NodeB", relation="connects_to")],
    )
    KnowledgeGraphStore.add_extracted_graph(tenant, extracted)

    json_str = KnowledgeGraphStore.export_to_json(tenant)
    assert "NodeA" in json_str
    assert "connects_to" in json_str

    # Restore in new tenant
    tenant_new = "test_graph_tenant_04_restored"
    KnowledgeGraphStore.load_from_json(tenant_new, json_str)
    g_restored = get_tenant_graph(tenant_new)

    assert g_restored.number_of_nodes() == 2
    assert g_restored.number_of_edges() == 1


# ── Graph Retriever Node Tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_graph_retriever_node():
    """Graph Retriever node extracts paths and adds them to AgentState."""
    tenant = "test_graph_tenant_05"
    g = get_tenant_graph(tenant)
    g.clear()

    extracted = ExtractedGraph(
        entities=[Entity(name="Falcon-H1R"), Entity(name="Mamba")],
        relationships=[Relationship(source="Falcon-H1R", target="Mamba", relation="combines_with")],
    )
    KnowledgeGraphStore.add_extracted_graph(tenant, extracted)

    state: AgentState = {
        "query": "How does Falcon-H1R use Mamba?",
        "tenant_id": tenant,
        "route_history": [],
    }

    res = await graph_retriever_node(state)
    assert len(res["graph_paths"]) >= 1
    assert res["graph_paths"][0]["source"] == "Falcon-H1R"
    assert "graph_retriever" in res["route_history"]


# ── Routing Tests ───────────────────────────────────────────────


def test_route_after_analyzer_graph_rag():
    """intent='graph_rag' runs the graph retriever alongside document retrieval.

    It used to route to graph_retriever alone. When the graph had no matching entity that
    produced an empty-evidence answer that the critic then escalated to web search — see
    tests/unit/test_source_routing.py for the full chain. The graph engine still runs; it
    just no longer excludes the user's documents.
    """
    state: AgentState = {"intent": "graph_rag"}
    assert route_after_analyzer(state) == "hybrid_multi_engine"


def test_route_after_rewriter_graph():
    """Query Rewriter with graph sub-query routes to graph_retriever."""
    state: AgentState = {
        "sub_queries": [{"query": "check relations", "source": "graph", "reasoning": ""}],
    }
    assert route_after_rewriter(state) == "graph_retriever"


def test_graph_compiles_with_graph_retriever():
    """StateGraph compiles successfully with Graph Retriever wired in."""
    graph = build_graph()
    compiled = graph.compile()
    assert compiled is not None
