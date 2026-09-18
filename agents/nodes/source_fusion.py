"""Source Fusion node — unifies internal chunks, graph relationships, and web results.

Merges heterogeneous evidence from:
1. Internal Document Chunks (with page, section, doc_id, rerank score)
2. Knowledge Graph Entity Paths (with multi-hop relationships and confidence)
3. Live Web Results (with title, URL, markdown snippet)

Produces a clean, 1-indexed list of FusedEvidence items ([^1], [^2], ...) that
the Synthesizer and Citation Critic use for precise citation grounding.
"""

from __future__ import annotations

import structlog

from agents.state import AgentState, FusedEvidence
from core.guardrails import sanitize_retrieved_content

logger = structlog.get_logger(__name__)

# Cross-encoder relevance below which a chunk is treated as "not actually about this".
# Both rerankers (Cohere rerank-v3.5 and the local FlashRank ms-marco cross-encoder)
# emit roughly 0–1 relevance. Measured against a known corpus, chunks that genuinely
# answered the question scored 0.59–0.98 while chunks retrieved purely as filler scored
# 0.006–0.037 — an order of magnitude apart, so this floor sits safely in the gap.
#
# Without it, a query the corpus cannot answer still hands the synthesizer whatever
# ranked highest and asks it to write. That burns a synthesis and a verification call
# to produce something the pipeline already had the evidence to refuse outright.
RELEVANCE_FLOOR = 0.05


async def source_fusion_node(state: AgentState) -> AgentState:
    """Fuse document chunks, knowledge graph paths, and web search results into a single indexed list."""
    route_history = list(state.get("route_history", []))
    route_history.append("source_fusion")

    chunks = state.get("retrieved_chunks", [])
    graph_paths = state.get("graph_paths", [])
    web_results = state.get("web_results", [])

    fused: list[FusedEvidence] = []
    source_idx = 1

    # Only chunks that were actually scored by a cross-encoder can be judged against
    # the floor — an un-reranked chunk carries a raw cosine score on a different scale,
    # so filtering those by the same number would discard good evidence.
    dropped_irrelevant = 0
    scored_chunks = []
    for chunk in chunks:
        if chunk.rerank_score is not None and chunk.rerank_score < RELEVANCE_FLOOR:
            dropped_irrelevant += 1
            continue
        scored_chunks.append(chunk)
    chunks = scored_chunks

    if dropped_irrelevant:
        logger.info(
            "Dropped chunks below the relevance floor",
            dropped=dropped_irrelevant,
            kept=len(chunks),
            floor=RELEVANCE_FLOOR,
        )

    # 1. Add internal document chunks
    injections_found = 0
    for chunk in chunks:
        doc_meta = chunk.metadata or {}
        section = doc_meta.get("heading") or doc_meta.get("section")
        page = doc_meta.get("page_number") or doc_meta.get("page")
        score = chunk.rerank_score if chunk.rerank_score is not None else chunk.dense_score

        doc_id = doc_meta.get("doc_id", "doc_unknown")
        title_parts = [f"Doc ID: {doc_id}"]
        if section:
            title_parts.append(f"Section: {section}")
        if page:
            title_parts.append(f"Page {page}")

        # A document's content is exactly as untrusted as a user's typed query — more
        # so, since anyone who ever got a file into the corpus authored it. Verified
        # directly: a delimiter-injection payload that gets BLOCKED when a user types
        # it sails through unfiltered here otherwise, and the synthesizer complies with
        # it, fabricating an unsupported claim under a real citation marker.
        safe_text, was_flagged = sanitize_retrieved_content(chunk.text)
        injections_found += was_flagged

        fused.append(
            {
                "source_id": source_idx,
                "marker": f"[^{source_idx}]",
                "source_type": "document",
                "title": " | ".join(title_parts),
                "content": safe_text,
                "url": None,
                "doc_id": doc_id,
                "chunk_id": chunk.id,
                "page": page,
                "section": section,
                "score": score,
            }
        )
        source_idx += 1

    # 2. Add knowledge graph entity relationship paths
    for path in graph_paths:
        path_str = path.get("path_str") or f"({path.get('source', '')}) ──[{path.get('relation', '')}]──► ({path.get('target', '')})"
        evidence = path.get("evidence", "")
        content = f"Relationship: {path_str}"
        if evidence:
            safe_evidence, was_flagged = sanitize_retrieved_content(evidence)
            injections_found += was_flagged
            content += f"\nEvidence: {safe_evidence}"

        fused.append(
            {
                "source_id": source_idx,
                "marker": f"[^{source_idx}]",
                "source_type": "graph",
                "title": f"Knowledge Graph Entity Link: {path.get('source')} ➔ {path.get('target')}",
                "content": content,
                "url": None,
                "doc_id": path.get("doc_id"),
                "chunk_id": path.get("chunk_id"),
                "page": None,
                "section": None,
                "score": path.get("confidence", 1.0),
            }
        )
        source_idx += 1

    # 3. Add external web results
    for web in web_results:
        url = web.get("url", "")
        title = web.get("title", "Web Source")
        snippet = web.get("snippet", "")

        if not snippet.strip():
            continue

        # Web content is at least as untrusted as an uploaded document — it's text
        # written by an arbitrary third party on the open internet.
        safe_snippet, was_flagged = sanitize_retrieved_content(snippet)
        injections_found += was_flagged

        fused.append(
            {
                "source_id": source_idx,
                "marker": f"[^{source_idx}]",
                "source_type": "web",
                "title": f"{title} ({url})" if url else title,
                "content": safe_snippet,
                "url": url,
                "doc_id": None,
                "chunk_id": None,
                "page": None,
                "section": None,
                "score": None,
            }
        )
        source_idx += 1

    if injections_found:
        logger.warning(
            "Redacted prompt-injection payload(s) from retrieved evidence before synthesis",
            count=injections_found,
        )

    logger.info(
        "Source fusion complete",
        total_fused_evidence=len(fused),
        document_count=len(chunks),
        graph_paths_count=len(graph_paths),
        web_count=len(web_results),
    )

    return {
        **state,
        "fused_evidence": fused,
        "route_history": route_history,
    }
