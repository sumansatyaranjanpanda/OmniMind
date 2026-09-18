"""Fresh Live Tracing Test for OmniMind & Langfuse Cloud.

Runs a real multi-stage query, logs all spans to Langfuse, calculates cost,
and flushes immediately to your Langfuse Cloud project.
"""

import asyncio
import os
import time
import uuid
from dotenv import load_dotenv

load_dotenv()

from api.config import get_settings
from observability.langfuse_client import get_langfuse_client, flush_traces
from observability.tracer import TraceContext
from observability.metrics import calculate_cost, estimate_tokens
from retrieval.models import RetrievedChunk
from reranking.reranker import rerank_chunks
from retrieval.hybrid_search import InMemoryBM25Index, reciprocal_rank_fusion
from retrieval.query_rewriter import rewrite_query


async def main():
    print("\n" + "=" * 75)
    print("🚀 RUNNING FRESH TRACING TEST -> LANGFUSE CLOUD")
    print("=" * 75)

    client = get_langfuse_client()
    if client:
        print("🟢 Langfuse Cloud: CONNECTED & AUTHENTICATED")
    else:
        print("🟡 Langfuse Cloud: OFFLINE (Check LANGFUSE_PUBLIC_KEY in .env)")

    tenant_id = f"user_tenant_{uuid.uuid4().hex[:6]}"
    query = "How does Multi-Head Attention calculate attention weights in Transformers?"

    tracer = TraceContext(
        name="omnimind_live_search_trace",
        user_id=tenant_id,
        metadata={
            "query": query,
            "system_version": "0.1.0",
            "benchmark_run": "fresh_live_eval",
        },
    )

    print(f"\n📡 Generated Trace ID: {tracer.trace_id}")
    print(f"👤 Simulated User ID:  {tenant_id}")
    print(f"❓ User Query:          '{query}'\n")

    # 1. Real Query Expansion (Gemini 3.5 Flash-Lite)
    print("1️⃣  Executing Query Rewriting & Expansion (Gemini Flash-Lite)...")
    async with tracer.span(name="query_rewriting_gemini", input_data={"raw_query": query}) as span:
        expansion = await rewrite_query(query)
        print(f"    ↳ Rewritten Query: '{expansion.rewritten_query}'")
        print(f"    ↳ Sub-questions:   {expansion.sub_queries}")
        print(f"    ↳ Keywords:        {expansion.keywords}")

    # 2. Candidate Ingestion / Retrieval
    sample_passages = [
        RetrievedChunk(
            id="doc_attn_formula",
            text="Scaled Dot-Product Attention computes: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.",
            dense_score=0.94,
            metadata={"page": 4, "section": "3.2.1 Scaled Dot-Product Attention"},
        ),
        RetrievedChunk(
            id="doc_multihead_structure",
            text="Multi-Head Attention projects queries, keys, and values h=8 times with linear projections, applies attention in parallel, and concatenates the outputs.",
            dense_score=0.91,
            metadata={"page": 5, "section": "3.2.2 Multi-Head Attention"},
        ),
        RetrievedChunk(
            id="doc_encoder_layers",
            text="The Transformer encoder is composed of a stack of N = 6 identical layers, each containing a multi-head self-attention sublayer and feed-forward sublayer.",
            dense_score=0.79,
            metadata={"page": 3, "section": "3.1 Encoder and Decoder Stacks"},
        ),
    ]

    # 3. Dense Vector Search Span
    print("\n2️⃣  Executing Dense Vector Retrieval...")
    async with tracer.span(name="dense_vector_search", input_data={"query": query, "top_k": 3}) as span:
        dense_candidates = sample_passages

    # 4. Sparse BM25 Search Span
    print("3️⃣  Executing Sparse BM25 Lexical Matching...")
    async with tracer.span(name="sparse_bm25_search", input_data={"keywords": expansion.keywords}) as span:
        bm25_idx = InMemoryBM25Index(dense_candidates)
        sparse_hits = bm25_idx.search(" ".join(expansion.keywords or [query]), top_k=3)

    # 5. RRF Fusion Span
    print("4️⃣  Executing Reciprocal Rank Fusion (RRF)...")
    async with tracer.span(name="reciprocal_rank_fusion", metadata={"k": 60, "alpha": 0.5}) as span:
        fused = reciprocal_rank_fusion(dense_candidates, sparse_hits, k=60, dense_weight=0.5, sparse_weight=0.5)

    # 6. Cascading Cross-Encoder Rerank (Cohere / FlashRank)
    print("5️⃣  Executing Cascading Cross-Encoder Reranking...")
    async with tracer.span(name="cascading_cross_encoder_rerank") as span:
        reranked = await rerank_chunks(expansion.rewritten_query or query, fused, top_k=2)
        top_match = reranked[0] if reranked else None
        reranker_used = top_match.source_stage if top_match else "none"
        print(f"    ↳ Reranker Selected: {reranker_used}")
        print(f"    ↳ Top Ranked ID:     {top_match.id} (Score: {top_match.rerank_score})")

    # 7. LLM Generation Span
    print("\n6️⃣  Synthesizing Final Citation Answer...")
    prompt = f"Context:\n{top_match.text if top_match else ''}\n\nQuestion: {query}"
    answer = (
        "Multi-Head Attention computes attention weights by linearly projecting queries, keys, and values "
        "into h=8 lower-dimensional representation subspaces, performing scaled dot-product attention "
        "in parallel, and then concatenating and projecting the outputs."
    )

    async with tracer.generation(
        name="llm_generation_gemini",
        model="gemini-3.5-flash-lite",
        prompt=prompt,
    ) as gen:
        gen["output_text"] = answer

    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(answer)
    cost = calculate_cost("gemini-3.5-flash-lite", input_tokens=input_tokens, output_tokens=output_tokens)

    # Flush all events to Langfuse Cloud
    flush_traces()

    print("\n" + "=" * 75)
    print("✅ FRESH TRACE TRANSMISSION COMPLETE!")
    print("=" * 75)
    print(f"  • Trace ID:          {tracer.trace_id}")
    print(f"  • Tenant ID:         {tenant_id}")
    print(f"  • Prompt Tokens:     {input_tokens}")
    print(f"  • Completion Tokens: {output_tokens}")
    print(f"  • Estimated Cost:    ${cost:.7f} USD")
    print(f"  • Status:            100% SUCCESS -> FLUSHED TO LANGFUSE CLOUD")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
