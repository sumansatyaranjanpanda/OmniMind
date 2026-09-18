"""LangGraph Adaptive Multi-Source Agent graph — the brain of OmniMind.

Assembles and compiles the full industry-standard Adaptive Agentic RAG workflow:

  START ──► cache_check ──► [hit?] ──► finalize_cached ──► END
                 │ (miss)
          query_analyzer
                 │
      ┌──────────┼─────────────────────┬──────────────────┬─────────────────┐
      ▼          ▼                     ▼                  ▼                 ▼
  direct_llm  retriever (vector)    graph_retriever   web_search       hybrid_multi_engine
      │          │                     │                  │                 │ (parallel)
      │          └──────────┬──────────┴──────────────────┴─────────────────┘
      │                     ▼
      │               source_fusion (unifies vectors + graph paths + web results)
      │                     ▼
      │                synthesizer
      │                     ▼
      │                  critic ──► [verified or max_retries?] ──► finalize ──► END
      │                                       │ (unverified retry)
      │                                 query_rewriter
      │                                       │ (targeted sub-queries)
      │                                       ▼
      │                          [route by retry source]
      └───────────────────────────────────────┼─────────────────────────────┘
                                              ▼
                                           finalize ──► END
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph

from agents.nodes.cache_check import cache_check_node, cache_store
from agents.nodes.context_rewriter import context_rewriter_node
from agents.nodes.critic import critic_node
from agents.nodes.direct_llm import direct_llm_node
from agents.nodes.graph_retriever import graph_retriever_node
from agents.nodes.guardrails import (
    finalize_blocked_node,
    input_guardrail_node,
    output_guardrail_node,
)
from agents.nodes.query_analyzer import query_analyzer_node
from agents.nodes.query_rewriter import query_rewriter_node
from agents.nodes.retriever import retriever_node
from agents.nodes.source_fusion import source_fusion_node
from agents.nodes.synthesizer import synthesizer_node
from agents.nodes.web_search import web_search_node
from agents.state import AgentState

logger = structlog.get_logger(__name__)


# ── Parallel Multi-Engine Hybrid Node ───────────────────────────


async def _skip_branch() -> None:
    """Placeholder for a branch that wasn't scheduled, so gather() always returns
    exactly three positional results and the merge below never has to guess which
    branch produced which slot."""
    return None


async def hybrid_multi_engine_node(state: AgentState) -> AgentState:
    """Execute internal vector retrieval, graph traversal, and web search in parallel.

    Each branch runs against the same pre-node `state`, so its returned dict is that
    state merged with only its own update — meaning every branch's result still carries
    every AgentState key, including the ones only a *different* branch populates. This
    used to merge by checking `"retrieved_chunks" in result`, which is true for every
    result regardless of which branch produced it, so whichever branch ran last in the
    list silently overwrote the earlier branches' real output with its own untouched
    (empty) copy of the same key. Because web_search always ran last, it clobbered the
    retriever's real chunks with `[]` on every query that also triggered a web branch —
    the internal retrieval was correct, it just never survived the merge. Reading each
    branch's own field by name instead of scanning for key presence removes the ambiguity.
    """
    route_history = list(state.get("route_history", []))
    route_history.append("hybrid_multi_engine")

    logger.info("Executing parallel multi-engine retrieval (vector + graph + web)")

    sub_queries = state.get("sub_queries", [])
    has_web = any(sq.get("source") == "web" for sq in sub_queries) or not sub_queries
    has_graph = any(sq.get("source") == "graph" for sq in sub_queries) or not sub_queries

    retriever_result, graph_result, web_result = await asyncio.gather(
        retriever_node(state),
        graph_retriever_node(state) if has_graph else _skip_branch(),
        web_search_node(state) if has_web else _skip_branch(),
    )

    return {
        **state,
        "retrieved_chunks": retriever_result.get("retrieved_chunks", []),
        "retrieval_trace_id": retriever_result.get("retrieval_trace_id"),
        "graph_paths": graph_result.get("graph_paths", []) if graph_result else [],
        "web_results": web_result.get("web_results", []) if web_result else [],
        "route_history": route_history,
    }


# How long to keep waiting for the query analyzer once the documents have already
# come back with usable evidence. Healthy analyzer calls measured 0.95-2.2s and have
# normally finished before retrieval does, so this only bites when the provider is
# throttling — which is exactly when not waiting matters most.
ANALYZER_GRACE_SECONDS = 1.5


async def analyze_and_retrieve_node(state: AgentState) -> AgentState:
    """Classify the query and search the documents at the same time.

    These were strictly sequential, and the ordering was costing real time for no
    benefit: measured live 2026-09-11, the analyzer took 2-3s and retrieval a further
    1-1.4s, so ~3.5s elapsed before synthesis could start — on a question the retriever
    could have begun answering immediately.

    Retrieval does not depend on the analyzer's output. It is safe to start early
    because of this file's own routing rule (see route_after_analyzer): every intent
    except pure small talk consults the documents anyway, so the speculative search is
    work we were always going to do. Small talk is detected without an LLM call by
    is_pleasantry, so we never start a search for "hi" in the first place.

    Net effect: the analyzer's latency is absorbed into retrieval's rather than added
    to it, and the classification it produces is still fully respected downstream.
    """
    from agents.nodes.query_analyzer import is_pleasantry
    from agents.nodes.source_fusion import RELEVANCE_FLOOR

    query = state.get("query", "").strip()

    # No documents worth searching for a greeting — and this path costs no LLM call,
    # so there is nothing to overlap it with.
    if is_pleasantry(query):
        return await query_analyzer_node(state)

    retrieval_task = asyncio.ensure_future(retriever_node(state))
    analyzer_task = asyncio.ensure_future(query_analyzer_node(state))

    try:
        retrieved = await retrieval_task
    except Exception as exc:
        # Speculation is an optimization, never a correctness dependency. If it failed,
        # let the normal retriever node run on whatever path the analyzer routes to.
        logger.warning("Speculative retrieval failed; retrying on the routed path", error=str(exc))
        retrieved = None
    except BaseException:
        analyzer_task.cancel()
        raise

    usable_evidence = [
        c
        for c in (retrieved or {}).get("retrieved_chunks", [])
        if c.rerank_score is None or c.rerank_score >= RELEVANCE_FLOOR
    ]

    # The documents already answered, so the analyzer is now only confirming a route we
    # are going to take regardless — and it is not worth an unbounded wait to hear it.
    # Measured live 2026-09-11: retrieval finished at 2.5s with a 0.85 top score while
    # the analyzer ran until 9.5s, then returned `internal_rag` with the query verbatim.
    # Seven seconds bought nothing.
    #
    # Degrading to `internal_rag` is specifically safe rather than merely convenient: it
    # is the documents-only route, which is exactly what this file's non-negotiable
    # prescribes as the default ("DOCUMENTS ARE ALWAYS CONSULTED"). The failure mode it
    # avoids — routing away from the corpus on a guess — is impossible here, because the
    # corpus has already been searched and has answered.
    #
    # When retrieval came back with nothing usable, the opposite holds: the analyzer's
    # graph-vs-web judgement is the whole decision, so it gets waited out in full.
    analyzed: AgentState | None = None
    if usable_evidence:
        try:
            analyzed = await asyncio.wait_for(analyzer_task, timeout=ANALYZER_GRACE_SECONDS)
        except asyncio.TimeoutError:
            logger.info(
                "Query analyzer exceeded its grace window; documents already answered, "
                "proceeding on the documents-only route",
                grace_seconds=ANALYZER_GRACE_SECONDS,
                evidence_count=len(usable_evidence),
            )
        except Exception as exc:
            logger.warning("Query analyzer failed; defaulting to documents-only", error=str(exc))
    else:
        try:
            analyzed = await analyzer_task
        except Exception as exc:
            logger.warning("Query analyzer failed with no evidence to fall back on", error=str(exc))

    if analyzed is None:
        analyzed = {
            **state,
            "intent": "internal_rag",
            "complexity": "simple",
            "sub_queries": [],
            "route_history": list(state.get("route_history", [])) + ["query_analyzer"],
        }

    # Small talk the heuristic didn't catch: the retrieval we ran is simply discarded.
    if analyzed.get("intent") == "direct_llm":
        return analyzed

    if retrieved is None:
        return analyzed

    internal_queries = [
        sq["query"] for sq in analyzed.get("sub_queries", []) if sq.get("source") == "internal"
    ] or [query]

    route_history = list(analyzed.get("route_history", []))
    searched = retrieved.get("speculative_retrieval_queries")
    if searched is None:
        searched = [query]

    # Only advertise the speculative result when it matches what the retriever would
    # now search. If the analyzer decomposed the question differently, retriever_node
    # falls through and searches properly — see the check there.
    if searched == internal_queries:
        route_history = list(retrieved.get("route_history", route_history))

    return {
        **analyzed,
        "retrieved_chunks": retrieved.get("retrieved_chunks", []),
        "retrieval_trace_id": retrieved.get("retrieval_trace_id"),
        "error": retrieved.get("error") or analyzed.get("error"),
        "speculative_retrieval_queries": searched,
        "route_history": route_history,
    }


# ── Conditional Routing Functions ───────────────────────────────


def route_after_input_guardrail(state: AgentState) -> str:
    """After input guardrail: if blocked, exit immediately; else proceed to cache check."""
    if state.get("guardrail_status") == "BLOCKED":
        return "finalize_blocked"
    return "cache_check"


def route_after_cache(state: AgentState) -> str:
    """After cache check: if hit, finalize immediately; else contextualize/rewrite query."""
    if state.get("cache_hit") and state.get("cached_answer"):
        return "finalize_cached"
    return "context_rewriter"


def route_after_analyzer(state: AgentState) -> str:
    """Route the query, with one hard rule: the user's documents always get consulted.

    This used to hand each intent to a single engine, and the non-internal ones bypassed
    document retrieval entirely. That made a pre-retrieval guess irreversible: "who is
    abinash" was classified graph_rag, the knowledge graph knew nothing about him, the
    critic failed the empty answer, the rewriter escalated to the web, and the user got a
    confident, correctly-cited summary of four unrelated strangers — while their own
    résumé sat in the index, never queried.

    A classifier cannot reliably know whether a corpus covers a subject; the retriever can
    simply check. So every intent except pure small talk now runs document retrieval, with
    graph/web engines running alongside it rather than instead of it. They execute in
    parallel, so consulting the documents costs no extra wall-clock time.
    """
    intent = state.get("intent", "internal_rag")

    if intent == "direct_llm":
        return "direct_llm"
    # graph_rag and web_search fan out to their engine AND the retriever, in parallel.
    if intent in ("graph_rag", "web_search", "hybrid"):
        return "hybrid_multi_engine"
    return "retriever"


def route_after_retriever(state: AgentState) -> str:
    """Documents came back — do we have anything worth synthesising from?

    Only reached on the documents-only path. If the corpus genuinely has nothing on the
    subject, fall back to the web rather than replying "insufficient evidence" — but mark
    it as a fallback so the answer can say where it came from.
    """
    from agents.nodes.source_fusion import RELEVANCE_FLOOR

    usable = [
        c
        for c in state.get("retrieved_chunks", [])
        if c.rerank_score is None or c.rerank_score >= RELEVANCE_FLOOR
    ]
    if usable:
        return "source_fusion"

    logger.info("No usable document evidence; falling back to web search")
    return "web_fallback"


def route_after_critic(state: AgentState) -> str:
    """After critic: if verified or max retries reached, sanitize and finalize; else rewrite query."""
    status = state.get("verification_status", "")
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)

    if status == "VERIFIED" or iteration >= max_iterations:
        return "output_guardrail"
    return "query_rewriter"


def route_after_rewriter(state: AgentState) -> str:
    """Route the re-search sub-queries to appropriate retrieval nodes."""
    sub_queries = state.get("sub_queries", [])
    has_internal = any(sq.get("source") == "internal" for sq in sub_queries)
    has_graph = any(sq.get("source") == "graph" for sq in sub_queries)
    has_web = any(sq.get("source") == "web" for sq in sub_queries)

    active_sources = sum([1 for flag in (has_internal, has_graph, has_web) if flag])
    if active_sources > 1:
        return "hybrid_multi_engine"
    if has_graph:
        return "graph_retriever"
    if has_web:
        return "web_search"
    return "retriever"


async def web_fallback_node(state: AgentState) -> AgentState:
    """Search the web because the user's own documents had nothing on this — or
    because retrieval couldn't check them at all.

    Three things matter here beyond running the search. First, the flag: the
    synthesizer uses it to tell the user the answer did not come from their documents,
    which is the disclosure that was missing when web results silently stood in for a
    document answer. Second, the query reset: whatever sub-queries the analyzer wrote
    were tuned for internal retrieval, so the web gets the user's actual question
    instead. Third — and this is the one that isn't obvious — WHY retrieval came back
    empty matters. Confirmed live with a simulated Pinecone outage: retriever_node sets
    state["error"] when every sub-query genuinely failed, but retrieved_chunks is still
    `[]` either way, so without checking `error` this path can't tell "the corpus was
    searched and had nothing" from "the corpus was never actually searched" — and told
    the user their own document didn't cover a topic it definitely does, because the
    search that would have found it never ran.
    """
    question = state.get("raw_query") or state.get("query", "")
    seeded: AgentState = {
        **state,
        "web_is_fallback": True,
        "retrieval_errored": bool(state.get("error")),
        "sub_queries": [
            {"query": question, "source": "web", "reasoning": "No document evidence found"}
        ],
    }
    return await web_search_node(seeded)


# ── Finalization Nodes ──────────────────────────────────────────


async def finalize_cached_node(state: AgentState) -> AgentState:
    """Finalize a cached response without running LLM synthesis."""
    route_history = list(state.get("route_history", []))
    route_history.append("finalize_cached")

    return {
        **state,
        "final_answer": state.get("cached_answer", ""),
        "verification_status": "CACHED",
        "faithfulness_score": 1.0,
        "route_history": route_history,
    }


async def finalize_node(state: AgentState) -> AgentState:
    """Finalize the verified answer and store it in the semantic cache."""
    route_history = list(state.get("route_history", []))
    route_history.append("finalize")

    final_answer = state.get("final_answer") or state.get("draft_answer", "")

    # Cache verified answers
    if final_answer and state.get("verification_status") in ("VERIFIED", "PARTIALLY_VERIFIED"):
        try:
            await cache_store(
                query=state["query"],
                answer=final_answer,
                tenant_id=state.get("tenant_id", "default"),
                citations=[dict(c) for c in state.get("citations", [])],
            )
        except Exception as e:
            logger.debug("Cache store failed (non-critical)", error=str(e))

    return {
        **state,
        "final_answer": final_answer,
        "route_history": route_history,
    }


# ── Graph Assembly ──────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct the state-of-the-art Adaptive Multi-Source state graph with Guardrails."""
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("finalize_blocked", finalize_blocked_node)
    graph.add_node("cache_check", cache_check_node)
    graph.add_node("finalize_cached", finalize_cached_node)
    graph.add_node("context_rewriter", context_rewriter_node)
    # Classification and document retrieval run concurrently inside this one node —
    # see analyze_and_retrieve_node for why that is safe.
    graph.add_node("query_analyzer", analyze_and_retrieve_node)
    graph.add_node("direct_llm", direct_llm_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("graph_retriever", graph_retriever_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("web_fallback", web_fallback_node)
    graph.add_node("hybrid_multi_engine", hybrid_multi_engine_node)
    graph.add_node("source_fusion", source_fusion_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("query_rewriter", query_rewriter_node)
    graph.add_node("output_guardrail", output_guardrail_node)
    graph.add_node("finalize", finalize_node)

    # Edges: START ──► input_guardrail
    graph.add_edge(START, "input_guardrail")

    # Conditional: input_guardrail ──► finalize_blocked OR cache_check
    graph.add_conditional_edges("input_guardrail", route_after_input_guardrail)

    # finalize_blocked ──► END
    graph.add_edge("finalize_blocked", END)

    # Conditional: cache_check ──► finalize_cached OR context_rewriter
    graph.add_conditional_edges("cache_check", route_after_cache)

    # finalize_cached ──► END
    graph.add_edge("finalize_cached", END)

    # context_rewriter ──► query_analyzer
    graph.add_edge("context_rewriter", "query_analyzer")

    # Conditional: query_analyzer ──► direct_llm | retriever | graph_retriever | web_search | hybrid_multi_engine
    graph.add_conditional_edges("query_analyzer", route_after_analyzer)

    # direct_llm ──► output_guardrail
    graph.add_edge("direct_llm", "output_guardrail")

    # retriever ──► source_fusion, or web_fallback when the corpus had nothing
    graph.add_conditional_edges("retriever", route_after_retriever)
    graph.add_edge("web_fallback", "source_fusion")

    # Remaining retrieval nodes ──► source_fusion
    graph.add_edge("graph_retriever", "source_fusion")
    graph.add_edge("web_search", "source_fusion")
    graph.add_edge("hybrid_multi_engine", "source_fusion")

    # source_fusion ──► synthesizer ──► critic
    graph.add_edge("source_fusion", "synthesizer")
    graph.add_edge("synthesizer", "critic")

    # Conditional: critic ──► output_guardrail OR query_rewriter (self-correction loop)
    graph.add_conditional_edges("critic", route_after_critic)

    # Conditional: query_rewriter ──► retriever | graph_retriever | web_search | hybrid_multi_engine
    graph.add_conditional_edges("query_rewriter", route_after_rewriter)

    # output_guardrail ──► finalize
    graph.add_edge("output_guardrail", "finalize")

    # finalize ──► END
    graph.add_edge("finalize", END)

    return graph


# ── Compiled Graph Singleton ────────────────────────────────────

_compiled_graph = None


def get_compiled_graph():
    """Get or create the compiled LangGraph agent singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_graph()
        _compiled_graph = graph.compile()
        logger.info("Adaptive Multi-Source Agent graph compiled successfully")
    return _compiled_graph


# ── Public API ──────────────────────────────────────────────────


async def run_agent(
    query: str,
    tenant_id: str,
    user_id: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    conversation_summary: str | None = None,
    thread_id: str | None = None,
    max_iterations: int = 2,
) -> dict[str, Any]:
    """Execute the full Adaptive Multi-Source agent pipeline.

    `chat_history` and `conversation_summary` are expected to already be loaded from
    Postgres by agents/memory/service.py — this function does not do that itself, so
    every caller (chat.py, chat_stream.py) must load context before invoking this.
    """
    compiled = get_compiled_graph()

    initial_state: AgentState = {
        "query": query,
        "raw_query": query,
        "rewritten_query": None,
        "thread_id": thread_id,
        "conversation_summary": conversation_summary,
        "user_id": user_id or tenant_id,
        "tenant_id": tenant_id,
        "chat_history": chat_history or [],
        "episodic_memories": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "route_history": [],
        "guardrail_status": "PASSED",
        "guardrail_reason": None,
        "masked_query": None,
        "pii_mapping": {},
        "cache_hit": False,
        "speculative_retrieval_queries": None,
        "intent": "",
        "complexity": "simple",
        "sub_queries": [],
        "is_direct_response": False,
        "retrieved_chunks": [],
        "graph_paths": [],
        "web_results": [],
        "web_is_fallback": False,
        "retrieval_errored": False,
        # Deliberately not seeded here. The analyzer fetches it (cached, 60s TTL) when the
        # key is absent; seeding an empty list would read as "this tenant has no documents"
        # and push every query to the web — the exact failure this change exists to fix.
        "fused_evidence": [],
        "citations": [],
        "faithfulness_score": 0.0,
        "verification_status": "",
        "critic_feedback": "",
        "unsupported_claims": [],
        "draft_answer": "",
        "final_answer": "",
        "error": None,
    }

    logger.info(
        "Starting Adaptive Multi-Source Agent",
        query=query,
        tenant_id=tenant_id,
        thread_id=thread_id,
        chat_history_len=len(chat_history or []),
        max_iterations=max_iterations,
    )

    # Execute the state graph
    final_state = await compiled.ainvoke(initial_state)

    logger.info(
        "Agent execution complete",
        route_history=final_state.get("route_history", []),
        intent=final_state.get("intent"),
        rewritten_query=final_state.get("rewritten_query"),
        verification_status=final_state.get("verification_status"),
        guardrail_status=final_state.get("guardrail_status"),
        faithfulness_score=final_state.get("faithfulness_score"),
        iterations=final_state.get("iteration", 0),
    )

    return {
        "answer": final_state.get("final_answer", ""),
        "citations": final_state.get("citations", []),
        "verification_status": final_state.get("verification_status", ""),
        "guardrail_status": final_state.get("guardrail_status", "PASSED"),
        "guardrail_reason": final_state.get("guardrail_reason"),
        "faithfulness_score": final_state.get("faithfulness_score", 0.0),
        "route_history": final_state.get("route_history", []),
        "intent": final_state.get("intent", "internal_rag"),
        "raw_query": final_state.get("raw_query", query),
        "rewritten_query": final_state.get("rewritten_query"),
        "thread_id": final_state.get("thread_id"),
        "retrieval_trace_id": final_state.get("retrieval_trace_id"),
        "cache_hit": final_state.get("cache_hit", False),
        "iteration_count": final_state.get("iteration", 0),
    }
