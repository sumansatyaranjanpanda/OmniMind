"""Synthetic Benchmark Dataset Generator for RAG Evaluation.

Automatically generates golden evaluation triples (Question, Ground Truth Answer, Contexts)
from document chunks to build automated regression test suites.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SynthesizedQAPair:
    """A single golden evaluation sample."""

    question: str
    ground_truth: str
    contexts: list[str]
    doc_id: str = "default_doc"
    chunk_id: str = "default_chunk"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DATASET_GENERATOR_SYSTEM_PROMPT = """You are an expert Test Dataset Engineer creating an automated benchmark for a RAG system.

Given a reference document chunk, generate a realistic, challenging user question and its definitive ground-truth answer based ONLY on the text.

Rules:
1. The question must test deep understanding (not just superficial keyword matching).
2. The ground_truth answer must be concise, accurate, and completely verifiable from the text.

Output ONLY valid JSON:
{
  "question": "<realistic user question>",
  "ground_truth": "<authoritative ground truth answer>"
}"""


async def generate_synthetic_qa(
    text: str,
    doc_id: str = "default_doc",
    chunk_id: str = "default_chunk",
) -> SynthesizedQAPair | None:
    """Generate a golden (Question, Ground Truth) sample from a text chunk."""
    clean_text = text.strip()
    if len(clean_text) < 30:
        return None

    try:
        from google import genai

        from api.config import get_settings

        settings = get_settings()
        client = genai.Client(api_key=settings.gemini_api_key)

        prompt = f"Reference Context:\n{clean_text}\n\nGenerate evaluation Q&A:"

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=DATASET_GENERATOR_SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )

        raw = response.text.strip()
        data = json.loads(raw)

        return SynthesizedQAPair(
            question=data.get("question", ""),
            ground_truth=data.get("ground_truth", ""),
            contexts=[clean_text],
            doc_id=doc_id,
            chunk_id=chunk_id,
        )

    except Exception as e:
        logger.warning("Synthetic QA generation failed; using heuristic template", error=str(e))
        # Heuristic fallback
        first_sentence = clean_text.split(".")[0]
        return SynthesizedQAPair(
            question=f"What does the document state about {first_sentence[:30]}?",
            ground_truth=first_sentence,
            contexts=[clean_text],
            doc_id=doc_id,
            chunk_id=chunk_id,
        )
