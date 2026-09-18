"""RAGAS Evaluation Metrics — quantitative quality guardrails for RAG.

Implements the standard 4 RAGAS metrics:
1. Faithfulness: Is the answer grounded in the retrieved context? (Hallucination detector)
2. Answer Relevance: Is the answer directly responsive to the user query?
3. Context Precision: Are the most relevant chunks ranked at the top? (Precision@k)
4. Context Recall: Does the context contain all facts present in the ground truth answer?
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RagasEvaluationResult:
    """Quantitative scores across all 4 RAGAS dimensions."""

    faithfulness: float  # 0.0 to 1.0
    answer_relevance: float  # 0.0 to 1.0
    context_precision: float  # 0.0 to 1.0
    context_recall: float  # 0.0 to 1.0
    overall_score: float  # Weighted mean

    def to_dict(self) -> dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "overall_score": round(self.overall_score, 4),
        }


# ── Metric Calculators ──────────────────────────────────────────


def calculate_context_precision(
    retrieved_contexts: list[str],
    ground_truth: str,
) -> float:
    """Calculate Context Precision @ k.

    Measures if ground-truth keywords/sentences appear at the highest ranking chunks.
    Formula: Mean Precision of relevant chunks at each rank k.
    """
    if not retrieved_contexts or not ground_truth:
        return 0.0

    gt_words = set(ground_truth.lower().split())
    if not gt_words:
        return 0.0

    relevant_count = 0
    precision_sum = 0.0

    for k, ctx in enumerate(retrieved_contexts, start=1):
        ctx_words = set(ctx.lower().split())
        overlap = len(gt_words.intersection(ctx_words)) / max(1, len(gt_words))

        # A chunk is considered relevant if it shares at least 20% key words with ground truth
        if overlap >= 0.20:
            relevant_count += 1
            precision_at_k = relevant_count / k
            precision_sum += precision_at_k

    if relevant_count == 0:
        return 0.0

    return round(precision_sum / relevant_count, 4)


def calculate_context_recall(
    retrieved_contexts: list[str],
    ground_truth: str,
) -> float:
    """Calculate Context Recall.

    Measures the ratio of key sentences/facts in the ground truth that can be
    found in the retrieved context.
    """
    if not retrieved_contexts or not ground_truth:
        return 0.0

    # Split ground truth into sentences/clauses
    gt_sentences = [s.strip() for s in ground_truth.replace("\n", ". ").split(". ") if len(s.strip()) > 10]
    if not gt_sentences:
        gt_sentences = [ground_truth]

    combined_context = " ".join(retrieved_contexts).lower()
    recalled_count = 0

    for sent in gt_sentences:
        words = [w for w in sent.lower().split() if len(w) > 3]
        if not words:
            continue
        matched_words = sum(1 for w in words if w in combined_context)
        if matched_words / len(words) >= 0.5:
            recalled_count += 1

    return round(recalled_count / max(1, len(gt_sentences)), 4)


async def evaluate_faithfulness(
    answer: str,
    retrieved_contexts: list[str],
) -> float:
    """LLM-assisted Faithfulness calculation.

    Extracts all claims from the answer and verifies what fraction is directly
    supported by the provided context.
    """
    if not answer or not retrieved_contexts:
        return 0.0

    try:
        from google import genai

        from api.config import get_settings

        settings = get_settings()
        client = genai.Client(api_key=settings.gemini_api_key)

        prompt = f"""Context:
{' '.join(retrieved_contexts[:5])}

Answer:
{answer}

Evaluate the faithfulness of this answer against the context.
Count:
1. Total factual claims made in the answer.
2. How many of those claims are strictly supported by the context.

Respond ONLY with valid JSON:
{{
  "total_claims": <int>,
  "supported_claims": <int>,
  "faithfulness_score": <float 0.0 to 1.0>
}}"""

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )

        raw = response.text.strip()
        data = json.loads(raw)
        return float(data.get("faithfulness_score", 0.85))

    except Exception as e:
        logger.debug("Faithfulness LLM calculation skipped; using token overlap", error=str(e))
        # Fallback token overlap
        ans_words = set(answer.lower().split())
        ctx_words = set(" ".join(retrieved_contexts).lower().split())
        overlap = len(ans_words.intersection(ctx_words)) / max(1, len(ans_words))
        return min(1.0, round(overlap * 1.2, 4))


async def evaluate_answer_relevance(
    query: str,
    answer: str,
) -> float:
    """Calculate Answer Relevance using embedding cosine similarity."""
    if not query or not answer:
        return 0.0

    try:
        from retrieval.pinecone_client import generate_text_embeddings

        embeddings = await generate_text_embeddings([query, answer])
        if len(embeddings) < 2:
            return 0.8

        # Cosine similarity
        vec_q = embeddings[0]
        vec_a = embeddings[1]
        dot = sum(q * a for q, a in zip(vec_q, vec_a))
        norm_q = math.sqrt(sum(q * q for q in vec_q))
        norm_a = math.sqrt(sum(a * a for a in vec_a))

        if norm_q == 0 or norm_a == 0:
            return 0.0

        cosine_sim = dot / (norm_q * norm_a)
        # Scale to [0.0, 1.0]
        return round(max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0)), 4)

    except Exception:
        # Fallback word overlap
        q_words = set(query.lower().split())
        a_words = set(answer.lower().split())
        return round(len(q_words.intersection(a_words)) / max(1, len(q_words)), 4)


async def evaluate_rag_sample(
    query: str,
    answer: str,
    retrieved_contexts: list[str],
    ground_truth: str,
) -> RagasEvaluationResult:
    """Compute all 4 RAGAS metrics for a single test sample."""
    import asyncio

    faith_task = evaluate_faithfulness(answer, retrieved_contexts)
    relevance_task = evaluate_answer_relevance(query, answer)

    faithfulness, relevance = await asyncio.gather(faith_task, relevance_task)
    precision = calculate_context_precision(retrieved_contexts, ground_truth)
    recall = calculate_context_recall(retrieved_contexts, ground_truth)

    overall = (faithfulness * 0.35) + (relevance * 0.25) + (precision * 0.20) + (recall * 0.20)

    return RagasEvaluationResult(
        faithfulness=faithfulness,
        answer_relevance=relevance,
        context_precision=precision,
        context_recall=recall,
        overall_score=overall,
    )
