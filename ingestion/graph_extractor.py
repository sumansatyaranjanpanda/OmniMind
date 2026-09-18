"""Knowledge Graph Extractor — transforms unstructured text chunks into entity-relation triplets.

Extracts:
1. Entities: Named concepts, algorithms, organizations, people, metrics, tools.
2. Relationships: Directed, typed edges connecting entities with evidence snippets.

Uses Gemini 3.5 Flash-Lite in structured JSON mode for high extraction precision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Entity:
    """A single named entity in the knowledge graph."""

    name: str
    entity_type: str = "CONCEPT"  # "CONCEPT" | "ALGORITHM" | "PERSON" | "ORGANIZATION" | "METRIC" | "TECHNOLOGY"
    description: str = ""


@dataclass
class Relationship:
    """A directed semantic relationship between two entities."""

    source: str
    target: str
    relation: str  # "uses" | "authored_by" | "component_of" | "outperforms" | "implements"
    confidence: float = 1.0
    evidence_snippet: str = ""
    chunk_id: str | None = None
    doc_id: str | None = None


@dataclass
class ExtractedGraph:
    """Container for all entities and relationships extracted from a chunk."""

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [asdict(e) for e in self.entities],
            "relationships": [asdict(r) for r in self.relationships],
        }


GRAPH_EXTRACTOR_SYSTEM_PROMPT = """You are an expert Knowledge Graph Information Extraction Agent for an enterprise AI system.

Your job: Given a text passage, extract all key named entities and the meaningful semantic relationships connecting them.

Entity Types:
- "CONCEPT" / "ALGORITHM": Mathematical equations, neural network components, algorithms (e.g., "Scaled Dot-Product Attention", "Softmax").
- "PERSON": Researchers, authors, executives (e.g., "Vaswani et al.", "Ashish Vaswani").
- "ORGANIZATION": Research labs, universities, companies (e.g., "Google Brain", "Google Research").
- "TECHNOLOGY" / "PRODUCT": Models, software frameworks, datasets (e.g., "Transformer", "PyTorch").
- "METRIC": Benchmarks, evaluation metrics (e.g., "BLEU Score", "Perplexity").

Relationship Types:
- "uses" / "contains" / "component_of"
- "authored_by" / "developed_by"
- "implements" / "based_on"
- "outperforms" / "evaluated_on"

Rules:
1. Normalize entity names to concise, standard title-case names (e.g., "Scaled Dot-Product Attention", not "the scaled dot product attention function").
2. Only extract relationships explicitly supported by the text.
3. Keep relationship predicates concise (lowercase verb phrases like "component_of", "uses", "authored_by").

Output ONLY valid JSON matching this schema:
{
  "entities": [
    {
      "name": "<normalized entity name>",
      "entity_type": "CONCEPT" | "ALGORITHM" | "PERSON" | "ORGANIZATION" | "TECHNOLOGY" | "METRIC",
      "description": "<brief 1-sentence definition>"
    }
  ],
  "relationships": [
    {
      "source": "<source entity name>",
      "target": "<target entity name>",
      "relation": "<directed relationship verb>",
      "confidence": <float 0.5 to 1.0>,
      "evidence_snippet": "<exact 1-sentence evidence from text>"
    }
  ]
}
"""


async def extract_knowledge_triplets(
    text: str,
    doc_id: str | None = None,
    chunk_id: str | None = None,
) -> ExtractedGraph:
    """Extract entities and semantic relationships from a text chunk."""
    clean_text = text.strip()
    if not clean_text or len(clean_text) < 20:
        return ExtractedGraph()

    try:
        from google import genai

        from api.config import get_settings

        settings = get_settings()
        client = genai.Client(api_key=settings.gemini_api_key)

        prompt = f"Text Passage:\n{clean_text}\n\nExtract knowledge graph entities and relationships:"

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=GRAPH_EXTRACTOR_SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        data = json.loads(raw_text)

        entities: list[Entity] = []
        for e in data.get("entities", []):
            if isinstance(e, dict) and e.get("name"):
                entities.append(
                    Entity(
                        name=e["name"].strip(),
                        entity_type=e.get("entity_type", "CONCEPT"),
                        description=e.get("description", ""),
                    )
                )

        relationships: list[Relationship] = []
        for r in data.get("relationships", []):
            if isinstance(r, dict) and r.get("source") and r.get("target"):
                relationships.append(
                    Relationship(
                        source=r["source"].strip(),
                        target=r["target"].strip(),
                        relation=r.get("relation", "relates_to").strip().lower().replace(" ", "_"),
                        confidence=float(r.get("confidence", 1.0)),
                        evidence_snippet=r.get("evidence_snippet", ""),
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                    )
                )

        logger.info(
            "Extracted knowledge graph elements",
            entities_count=len(entities),
            relationships_count=len(relationships),
            doc_id=doc_id,
            chunk_id=chunk_id,
        )

        return ExtractedGraph(entities=entities, relationships=relationships)

    except Exception as e:
        logger.warning("Knowledge graph extraction failed; falling back to heuristic", error=str(e))
        return _heuristic_extraction(clean_text, doc_id, chunk_id)


def _heuristic_extraction(text: str, doc_id: str | None, chunk_id: str | None) -> ExtractedGraph:
    """Fast rule-based heuristic extraction fallback when LLM is unavailable."""
    entities: list[Entity] = []
    relationships: list[Relationship] = []

    # Common AI concepts heuristic
    key_terms = ["Transformer", "Attention", "Scaled Dot-Product Attention", "Multi-Head Attention", "Softmax", "Vaswani"]
    found_terms = [t for t in key_terms if t.lower() in text.lower()]

    for term in found_terms:
        entities.append(Entity(name=term, entity_type="CONCEPT", description="Key architecture component"))

    if "Transformer" in found_terms and "Attention" in found_terms:
        relationships.append(
            Relationship(
                source="Transformer",
                target="Attention",
                relation="uses",
                confidence=0.9,
                evidence_snippet=text[:100],
                doc_id=doc_id,
                chunk_id=chunk_id,
            )
        )

    return ExtractedGraph(entities=entities, relationships=relationships)
