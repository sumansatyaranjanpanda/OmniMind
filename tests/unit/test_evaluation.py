"""Unit tests for Phase 6 RAGAS Evaluation Metrics and Synthetic Dataset Generation."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from evaluation.dataset_generator import SynthesizedQAPair, generate_synthetic_qa
from evaluation.metrics import (
    RagasEvaluationResult,
    calculate_context_precision,
    calculate_context_recall,
    evaluate_answer_relevance,
    evaluate_faithfulness,
    evaluate_rag_sample,
)


def test_context_precision_perfect_match():
    """Context precision is high when the most relevant chunk is at index 1."""
    contexts = [
        "Scaled Dot-Product Attention computes softmax(QK^T / sqrt(d_k)) * V on packed queries and keys.",
        "Some unrelated background information about deep learning.",
    ]
    gt = "Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V"

    precision = calculate_context_precision(contexts, gt)
    assert precision >= 0.80


def test_context_precision_empty():
    """Context precision handles empty inputs safely."""
    assert calculate_context_precision([], "some ground truth") == 0.0
    assert calculate_context_precision(["some context"], "") == 0.0


def test_context_recall():
    """Context recall measures what fraction of ground truth statements exist in context."""
    contexts = [
        "The model uses multi-head attention. We compute the dot product of queries and keys.",
        "Dimension d_k is 64 and dimension d_v is 64.",
    ]
    gt = "The model uses multi-head attention. Queries and keys have dimension d_k."

    recall = calculate_context_recall(contexts, gt)
    assert recall >= 0.80


def test_context_recall_empty():
    """Context recall returns 0.0 for empty inputs."""
    assert calculate_context_recall([], "some fact") == 0.0
    assert calculate_context_recall(["some context"], "") == 0.0


@pytest.mark.asyncio
async def test_evaluate_faithfulness_fallback():
    """Faithfulness calculation falls back gracefully without LLM."""
    answer = "The Attention equation uses softmax and sqrt(d_k)."
    contexts = ["Attention equation uses softmax and sqrt(d_k) for scaling."]

    score = await evaluate_faithfulness(answer, contexts)
    assert 0.0 <= score <= 1.0
    assert score >= 0.70


@pytest.mark.asyncio
async def test_evaluate_answer_relevance():
    """Answer relevance calculates a similarity score between 0.0 and 1.0."""
    query = "What is attention?"
    answer = "Attention is a mechanism that computes softmax weights over key value pairs."

    with patch("retrieval.pinecone_client.generate_text_embeddings", AsyncMock(return_value=[[0.1] * 256, [0.2] * 256])):
        score = await evaluate_answer_relevance(query, answer)
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_evaluate_rag_sample():
    """Full sample evaluation returns a complete RagasEvaluationResult."""
    query = "What is scaled dot product attention?"
    answer = "It computes softmax(QK^T / sqrt(d_k)) * V."
    contexts = ["Scaled dot product attention formula: softmax(QK^T / sqrt(d_k)) * V."]
    gt = "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V."

    with patch("retrieval.pinecone_client.generate_text_embeddings", AsyncMock(return_value=[[0.1] * 256, [0.2] * 256])):
        result: RagasEvaluationResult = await evaluate_rag_sample(query, answer, contexts, gt)
    assert 0.0 <= result.faithfulness <= 1.0
    assert 0.0 <= result.answer_relevance <= 1.0
    assert 0.0 <= result.context_precision <= 1.0
    assert 0.0 <= result.context_recall <= 1.0
    assert 0.0 <= result.overall_score <= 1.0


@pytest.mark.asyncio
async def test_synthetic_qa_generation():
    """Synthetic QA generator creates valid Q&A pair."""
    text = "The Transformer architecture was introduced by Vaswani et al. in 2017 to replace recurrent networks."
    sample = await generate_synthetic_qa(text, doc_id="doc_test", chunk_id="chk_test")

    assert sample is not None
    assert isinstance(sample, SynthesizedQAPair)
    assert len(sample.question) > 5
    assert len(sample.ground_truth) > 5
    assert sample.doc_id == "doc_test"
