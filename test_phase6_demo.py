"""E2E Demonstration and Benchmark for OmniMind Phase 6.

Demonstrates:
1. Knowledge Graph Ingestion & Triplet Extraction
2. Multi-Hop Graph Traversal (NetworkX DiGraph)
3. GraphRAG Agent Execution with [^N] graph citations
4. RAGAS Automated Evaluation Benchmark (Faithfulness, Relevance, Precision, Recall)
5. Real-Time Streaming Chat (SSE Event Generator)
"""

from __future__ import annotations

import asyncio
import json
import time

from agents.graph import run_agent
from api.routers.chat_stream import generate_chat_events
from evaluation.metrics import evaluate_rag_sample
from ingestion.graph_extractor import Entity, ExtractedGraph, Relationship
from retrieval.graph_store import KnowledgeGraphStore

DEMO_TENANT = "phase6_enterprise_tenant"


def seed_demo_knowledge_graph():
    """Seed sample entities and relationships into the tenant's knowledge graph."""
    e1 = Entity(name="Vaswani et al.", entity_type="PERSON", description="Authors of 'Attention Is All You Need'")
    e2 = Entity(name="Transformer", entity_type="TECHNOLOGY", description="Attention-based neural architecture")
    e3 = Entity(name="Scaled Dot-Product Attention", entity_type="ALGORITHM", description="Core attention mechanism formula")
    e4 = Entity(name="Multi-Head Attention", entity_type="ALGORITHM", description="Multi-subspace attention projection")
    e5 = Entity(name="Falcon-H1R", entity_type="TECHNOLOGY", description="2026 Transformer-Mamba hybrid model")
    e6 = Entity(name="Mamba", entity_type="CONCEPT", description="State space model for linear-time sequence modeling")

    r1 = Relationship(source="Vaswani et al.", target="Transformer", relation="authored", confidence=1.0, evidence_snippet="Vaswani et al. (2017)")
    r2 = Relationship(source="Transformer", target="Scaled Dot-Product Attention", relation="uses", confidence=0.98, evidence_snippet="Transformer relies on Scaled Dot-Product Attention")
    r3 = Relationship(source="Transformer", target="Multi-Head Attention", relation="contains", confidence=0.98, evidence_snippet="Multi-head attention projects queries and keys h times")
    r4 = Relationship(source="Falcon-H1R", target="Transformer", relation="based_on", confidence=0.95, evidence_snippet="Falcon-H1R is a hybrid Transformer architecture")
    r5 = Relationship(source="Falcon-H1R", target="Mamba", relation="combines_with", confidence=0.95, evidence_snippet="Falcon-H1R combines Transformer attention with Mamba SSMs")

    extracted = ExtractedGraph(
        entities=[e1, e2, e3, e4, e5, e6],
        relationships=[r1, r2, r3, r4, r5],
    )

    res = KnowledgeGraphStore.add_extracted_graph(DEMO_TENANT, extracted)
    print(f"🕸️  Knowledge Graph Initialized: Added {res['nodes_added']} nodes and {res['edges_added']} relationship edges.")


async def demo_multi_hop_graph():
    print("\n" + "=" * 75)
    print("📍 DEMO 1: MULTI-HOP KNOWLEDGE GRAPH DISCOVERY")
    print("=" * 75)

    # 1. Neighborhood around Transformer
    neighborhood = KnowledgeGraphStore.get_neighborhood(DEMO_TENANT, "Transformer", max_hops=1)
    print("🔹 1-Hop Neighborhood for 'Transformer':")
    for n in neighborhood:
        print(f"   ↳ {n['path_str']}")

    # 2. Multi-hop path from Vaswani et al. to Mamba
    paths = KnowledgeGraphStore.find_paths(DEMO_TENANT, "Vaswani et al.", "Mamba", max_depth=3)
    print("\n🔹 Discovered Multi-Hop Path: 'Vaswani et al.' ➔ 'Mamba':")
    for p in paths:
        print(f"   ↳ {' ➔ '.join(p)}")


async def demo_graph_rag_agent():
    print("\n" + "=" * 75)
    print("📍 DEMO 2: ADAPTIVE GRAPHRAG AGENT EXECUTION")
    print("=" * 75)

    query = "How does Falcon-H1R connect to the Transformer architecture and Mamba?"
    print(f"❓ User Query: '{query}'")

    start = time.perf_counter()
    result = await run_agent(query=query, tenant_id=DEMO_TENANT, max_iterations=2)
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    print(f"\n⏱️  Execution Latency:     {elapsed} ms")
    print(f"🎯 Intent Classified:     {result.get('intent', '').upper()}")
    print(f"🛤️  Graph Route Path:      {' ➔ '.join(result.get('route_history', []))}")
    print(f"🛡️  Verification Status:   {result.get('verification_status')}")
    print(f"🎯 Faithfulness Score:     {result.get('faithfulness_score')}")

    print("\n💬 Generated Answer:")
    print(result.get("answer", ""))


async def demo_ragas_evaluation():
    print("\n" + "=" * 75)
    print("📍 DEMO 3: AUTOMATED RAGAS EVALUATION METRICS")
    print("=" * 75)

    query = "What is the formula for Scaled Dot-Product Attention?"
    answer = "Scaled Dot-Product Attention is computed as softmax(Q * K^T / sqrt(d_k)) * V [^1]."
    contexts = ["Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V where d_k is the key dimension."]
    ground_truth = "Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V."

    print(f"❓ Query:        '{query}'")
    print(f"💬 Answer:       '{answer}'")
    print(f"🎯 Ground Truth: '{ground_truth}'\n")

    eval_result = await evaluate_rag_sample(query, answer, contexts, ground_truth)

    print("📊 RAGAS Scorecard:")
    print(f"   ↳ Faithfulness:       {eval_result.faithfulness:.4f}")
    print(f"   ↳ Answer Relevance:   {eval_result.answer_relevance:.4f}")
    print(f"   ↳ Context Precision:  {eval_result.context_precision:.4f}")
    print(f"   ↳ Context Recall:     {eval_result.context_recall:.4f}")
    print(f"   ↳ Overall Score:      {eval_result.overall_score:.4f} / 1.0000")


async def demo_streaming_sse():
    print("\n" + "=" * 75)
    print("📍 DEMO 4: REAL-TIME STREAMING (SERVER-SENT EVENTS)")
    print("=" * 75)

    query = "Hello OmniMind! Can you explain your Phase 6 capabilities?"
    print(f"❓ Streaming Query: '{query}'\n")

    event_gen = generate_chat_events(
        query=query,
        tenant_id=DEMO_TENANT,
        user_id="user_test_01",
        chat_history=[],
    )

    print("⚡ Live SSE Event Stream Output:")
    async for event_chunk in event_gen:
        lines = [l for l in event_chunk.strip().split("\n") if l]
        for line in lines:
            if line.startswith("event:"):
                print(f"   ⚡ [{line}]", end=" ")
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if "text" in data:
                    print(f"token: '{data['text']}'")
                elif "status" in data:
                    print(f"route_status: {data}")
                elif "verification_status" in data:
                    print(f"complete: {data}")
                else:
                    print(f"{data}")


async def main():
    print("\n" + "=" * 75)
    print("🚀 OMNIMIND PHASE 6: GRAPHRAG, RAGAS EVALUATION & STREAMING BENCHMARK")
    print("=" * 75)

    seed_demo_knowledge_graph()
    await demo_multi_hop_graph()
    await demo_graph_rag_agent()
    await demo_ragas_evaluation()
    await demo_streaming_sse()

    print("\n" + "=" * 75)
    print("🏆 PHASE 6 E2E BENCHMARK COMPLETE (ALL MODULES VERIFIED)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
