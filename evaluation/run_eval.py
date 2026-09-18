"""Automated RAGAS Evaluation Benchmark Runner for OmniMind.

Executes end-to-end evaluation runs on test datasets, computing:
1. Average Faithfulness
2. Average Answer Relevance
3. Average Context Precision
4. Average Context Recall
5. Average Latency (ms)
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import structlog

from agents.graph import run_agent
from evaluation.metrics import RagasEvaluationResult, evaluate_rag_sample

logger = structlog.get_logger(__name__)

BENCHMARK_SAMPLES = [
    {
        "query": "What is the mathematical equation for Scaled Dot-Product Attention?",
        "ground_truth": "Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V where Q and K have dimension d_k and V has dimension d_v.",
        "tenant_id": "demo_enterprise_tenant_01",
    },
    {
        "query": "How is Multi-Head Attention computed in the Transformer model?",
        "ground_truth": "MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O, where each head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V).",
        "tenant_id": "demo_enterprise_tenant_01",
    },
]


async def run_benchmark(
    samples: list[dict[str, Any]] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run full evaluation suite across benchmark samples."""
    test_samples = samples or BENCHMARK_SAMPLES
    results: list[dict[str, Any]] = []

    print("\n" + "=" * 75)
    print("📊 STARTING OMNIMIND AUTOMATED RAGAS EVALUATION BENCHMARK")
    print("=" * 75)

    for i, sample in enumerate(test_samples, start=1):
        query = sample["query"]
        ground_truth = sample["ground_truth"]
        tenant_id = sample.get("tenant_id", "demo_enterprise_tenant_01")

        print(f"\n[{i}/{len(test_samples)}] Evaluating: '{query}'")
        start = time.perf_counter()

        # 1. Run agent pipeline
        agent_out = await run_agent(query=query, tenant_id=tenant_id)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        answer = agent_out.get("answer", "")
        # Extract retrieved context texts from citations or chunks
        citations = agent_out.get("citations", [])
        contexts = [c.get("text_snippet", "") for c in citations if c.get("text_snippet")]
        if not contexts:
            contexts = [answer]

        # 2. Compute RAGAS metrics
        metrics: RagasEvaluationResult = await evaluate_rag_sample(
            query=query,
            answer=answer,
            retrieved_contexts=contexts,
            ground_truth=ground_truth,
        )

        sample_res = {
            "query": query,
            "latency_ms": latency_ms,
            "intent": agent_out.get("intent"),
            "verification_status": agent_out.get("verification_status"),
            "metrics": metrics.to_dict(),
        }
        results.append(sample_res)

        print(f"   ↳ Latency:            {latency_ms} ms")
        print(f"   ↳ Faithfulness:       {metrics.faithfulness:.4f}")
        print(f"   ↳ Answer Relevance:   {metrics.answer_relevance:.4f}")
        print(f"   ↳ Context Precision:  {metrics.context_precision:.4f}")
        print(f"   ↳ Context Recall:     {metrics.context_recall:.4f}")
        print(f"   ↳ Overall Score:      {metrics.overall_score:.4f}")

    # Compute aggregates
    avg_faithfulness = sum(r["metrics"]["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["metrics"]["answer_relevance"] for r in results) / len(results)
    avg_precision = sum(r["metrics"]["context_precision"] for r in results) / len(results)
    avg_recall = sum(r["metrics"]["context_recall"] for r in results) / len(results)
    avg_overall = sum(r["metrics"]["overall_score"] for r in results) / len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    summary = {
        "timestamp": time.time(),
        "total_samples": len(results),
        "avg_latency_ms": round(avg_latency, 2),
        "avg_faithfulness": round(avg_faithfulness, 4),
        "avg_answer_relevance": round(avg_relevance, 4),
        "avg_context_precision": round(avg_precision, 4),
        "avg_context_recall": round(avg_recall, 4),
        "avg_overall_ragas_score": round(avg_overall, 4),
        "samples": results,
    }

    print("\n" + "=" * 75)
    print("🏆 RAGAS BENCHMARK SCORECARD SUMMARY")
    print("=" * 75)
    print(f"📈 Average Overall RAGAS Score:  {summary['avg_overall_ragas_score']:.4f} / 1.0000")
    print(f"🎯 Average Faithfulness:         {summary['avg_faithfulness']:.4f}")
    print(f"💡 Average Answer Relevance:     {summary['avg_answer_relevance']:.4f}")
    print(f"🔍 Average Context Precision:    {summary['avg_context_precision']:.4f}")
    print(f"📚 Average Context Recall:       {summary['avg_context_recall']:.4f}")
    print(f"⏱️  Average Pipeline Latency:    {summary['avg_latency_ms']:.2f} ms")
    print("=" * 75 + "\n")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"💾 Report saved to: {output_path}")

    return summary


if __name__ == "__main__":
    asyncio.run(run_benchmark())
