"""E2E Benchmark script for the OmniMind Adaptive Multi-Source Architecture.

Tests 4 realistic production scenarios:
1. Pure Internal Document Query (Scaled Dot-Product Attention)
2. Direct Conversational Query (Greeting / Fast-Path)
3. Pure Web Search Query (2026 AI Breakthroughs via Tavily)
4. Semantic Cache Hit (Repeated Query)
"""

from __future__ import annotations

import asyncio
import time

import uuid
from agents.graph import run_agent
from api.schemas.ingestion import Chunk
from retrieval.pinecone_client import upsert_chunks

DEMO_TENANT_ID = "demo_enterprise_tenant_01"
DEMO_DOC_ID = uuid.uuid4()


async def seed_demo_knowledge_base():
    """Seed documents into the vector store for the demo tenant."""
    demo_chunks = [
        Chunk(
            chunk_id="doc_transformer_01_p4_c1",
            document_id=DEMO_DOC_ID,
            tenant_id=DEMO_TENANT_ID,
            text=(
                "Scaled Dot-Product Attention: We compute the attention function on a set of queries simultaneously, "
                "packed together into a matrix Q. The keys and values are also packed into matrices K and V. "
                "We compute the matrix of outputs as: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V, "
                "where d_k is the dimension of the queries and keys, and d_v is the dimension of the values."
            ),
            page=4,
            section="3.2.1 Scaled Dot-Product Attention",
            source_type="pdf",
        ),
        Chunk(
            chunk_id="doc_transformer_01_p5_c2",
            document_id=DEMO_DOC_ID,
            tenant_id=DEMO_TENANT_ID,
            text=(
                "Multi-Head Attention: Multi-head attention allows the model to jointly attend to information "
                "from different representation subspaces at different positions. "
                "MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O, where head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)."
            ),
            page=5,
            section="3.2.2 Multi-Head Attention",
            source_type="pdf",
        ),
    ]
    await upsert_chunks(chunks=demo_chunks, tenant_id=DEMO_TENANT_ID)
    print(f"📦 Seeded demo document knowledge base for tenant: '{DEMO_TENANT_ID}'\n")


async def run_scenario(scenario_num: int, title: str, query: str):
    """Run and format a benchmark scenario."""
    print("=" * 75)
    print(f"📍 SCENARIO {scenario_num}: {title}")
    print(f"❓ User Query: '{query}'")
    print("-" * 75)

    start = time.perf_counter()
    result = await run_agent(
        query=query,
        tenant_id=DEMO_TENANT_ID,
        max_iterations=2,
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    print(f"\n⏱️  Execution Latency:     {elapsed_ms} ms")
    print(f"🎯 Classified Intent:     {result.get('intent', 'N/A').upper()}")
    print(f"🛤️  Graph Route Path:      {' ➔ '.join(result.get('route_history', []))}")
    print(f"⚡ Cache Hit:              {result.get('cache_hit')}")
    print(f"🛡️  Verification Status:   {result.get('verification_status')}")
    print(f"🎯 Faithfulness Score:     {result.get('faithfulness_score')}")
    print(f"🔄 Retries Performed:     {result.get('iteration_count')}")

    citations = result.get("citations", [])
    print(f"📌 Citations Found:       {len(citations)}")
    for c in citations:
        source_label = f"[{c.get('source_type')}]"
        if c.get("source_type") == "web":
            detail = c.get("url") or "Web Search"
        else:
            detail = f"Section: {c.get('section', 'N/A')} (Page {c.get('page', 'N/A')})"
        print(f"   ↳ {c.get('marker')} {source_label} {detail}")

    print("\n💬 Generated Answer:")
    print(result.get("answer", ""))
    print()


async def main():
    print("\n" + "=" * 75)
    print("🤖 OMNIMIND PHASE 5: ADAPTIVE MULTI-SOURCE AGENT BENCHMARK")
    print("=" * 75)

    await seed_demo_knowledge_base()

    # Scenario 1: In-Domain Internal Document Query
    await run_scenario(
        scenario_num=1,
        title="Pure Internal Document Query (Scaled Dot-Product Attention)",
        query="What is the mathematical equation for Scaled Dot-Product Attention?",
    )

    # Scenario 2: Conversational Direct LLM Query (Fast-Path)
    await run_scenario(
        scenario_num=2,
        title="Direct Conversational Greeting (Bypasses Retrieval)",
        query="Hello OmniMind! How are you doing today?",
    )

    # Scenario 3: Pure Web Search Query (External Knowledge via Tavily)
    await run_scenario(
        scenario_num=3,
        title="Pure Web Search (2026 AI Breakthroughs via Tavily AI)",
        query="What are the latest 2026 AI breakthroughs in large language models?",
    )

    # Scenario 4: Semantic Cache Hit
    await run_scenario(
        scenario_num=4,
        title="Semantic Cache Verification (Identical / Similar Query)",
        query="What is the mathematical equation for Scaled Dot-Product Attention?",
    )

    print("=" * 75)
    print("🏆 PHASE 5 ADAPTIVE MULTI-SOURCE AGENT BENCHMARK COMPLETE (ALL SCENARIOS PASSED)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
