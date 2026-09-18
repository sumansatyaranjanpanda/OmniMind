"""Query Analyzer node — pre-retrieval intent classification and query decomposition.

Acts as the intelligent front-door router:
1. Classifies query intent:
   - "direct_llm": Greetings, conversational pleasantries, simple non-factual chats.
   - "internal_rag": Domain-specific technical questions, company docs, proprietary topics.
   - "graph_rag": Questions about entity relationships, authors, multi-hop dependencies, hierarchies.
   - "web_search": Real-time info, breaking news, external product comparisons, live prices.
   - "hybrid": Questions combining multiple sources (e.g. internal docs + graph + web).
2. Decomposes complex / multi-part questions into focused sub-queries, each explicitly
   assigned to internal document retriever, knowledge graph retriever, or live web search.
"""

from __future__ import annotations

import json
import re

import structlog

from agents.state import AgentState, SubQuery

logger = structlog.get_logger(__name__)

# Conversational pleasantries bypass LLM query analysis entirely.
#
# This was one anchored mega-regex that required the trigger word to be followed by
# whitespace, so any attached punctuation broke it — "Thanks, that's helpful. Have a
# good day!" failed to match, fell through to the LLM classifier, timed out, and hit
# the internal_rag fallback, which ran full retrieval plus a web search and took 70
# seconds to say "you're welcome". Splitting the decision into three named parts is
# both easier to reason about and easier to extend than growing that regex further.

_COURTESY_OPENER_RE = re.compile(
    r"^(?:hi|hello|hey|greetings|howdy|yo|thanks|thank\s+you|thx|cheers|"
    r"good\s+(?:morning|afternoon|evening|day)|bye|goodbye|see\s+you|"
    r"have\s+a\s+(?:good|great|nice)\s+\w+)\b",
    re.IGNORECASE,
)

# Words that signal a real information request rather than small talk.
_INTERROGATIVE_RE = re.compile(
    r"\b(?:what|how|why|when|where|which|whose|can|could|would|do|does|did|is|are|was|were|"
    r"should|list|show|find|give|tell|explain|compare|summari[sz]e)\b",
    re.IGNORECASE,
)

# "who are you" / "what are you" are about the assistant itself — always direct_llm,
# even though they trip the interrogative check above.
_IDENTITY_RE = re.compile(r"^(?:who|what)\s+(?:are|is)\s+you\b", re.IGNORECASE)


def is_pleasantry(query: str) -> bool:
    """True when the query is small talk that needs no retrieval.

    Deliberately conservative — a real question that merely opens with a greeting
    ("Hi, what is our PTO policy?") must NOT match, because answering it from the
    model alone would silently skip the user's documents.
    """
    text = " ".join(query.split())
    if not text:
        return False
    if _IDENTITY_RE.match(text):
        return True
    opener = _COURTESY_OPENER_RE.match(text)
    if not opener:
        return False

    remainder = text[opener.end():]
    if not re.search(r"[a-zA-Z0-9]", remainder):
        return True  # bare "hi", "thanks!", "good morning."
    if "?" in remainder or _INTERROGATIVE_RE.search(remainder) or re.search(r"\d", remainder):
        return False  # a real question wearing a polite hat
    return len(remainder.split()) <= 8


def looks_like_a_question(query: str) -> bool:
    """Cheap check used only to pick a fallback route when classification fails."""
    return bool("?" in query or _INTERROGATIVE_RE.search(query))

QUERY_ANALYZER_SYSTEM_PROMPT = """You are an expert Query Analysis and Routing Agent for a document-question-answering system.

The user has uploaded their own documents. Your job is to analyze their query and produce
a structured execution plan.

THE GOVERNING RULE: the user's own documents are the reason this product exists. If the
uploaded documents could plausibly contain the answer, route to "internal_rag". Only choose
"web_search" when the question is clearly about something no uploaded document could cover.
When you are unsure, choose "internal_rag" — searching the user's documents and finding
nothing is cheap and harmless, while answering from the public web about a subject that
lives in their documents produces confidently wrong answers about the wrong subject.

Intent Taxonomy:
- "direct_llm": Greetings, thanks, small talk, or questions about you the assistant.
  NOT for any question seeking information.
- "internal_rag": THE DEFAULT for any information-seeking question. Anything a document
  might answer — including questions about people, companies, projects, products, dates,
  numbers, policies, or summaries. A question about a named person or organisation is
  internal_rag whenever the corpus might describe them.
- "graph_rag": Only for explicit relationship/multi-hop questions ("which component depends
  on which", "how are X and Y connected"). A simple "who is X" is internal_rag, NOT graph_rag.
- "web_search": Only for information that is inherently external and current — live prices,
  today's news, weather, sports results, or an explicitly public entity the documents
  clearly do not cover. If the user's document list mentions the subject, this is wrong.
- "hybrid": The question needs the user's documents AND live external context together
  (e.g. "how does our pricing compare to competitors right now").

Complexity Taxonomy:
- "simple": A single factual question.
- "multi_hop": A complex question with multiple parts or requiring comparative reasoning across sources.

Decomposition Rules:
- For "simple" internal_rag: produce 1 sub-query targeting "internal".
- For "simple" graph_rag: produce 1 sub-query targeting "graph".
- For "simple" web_search: produce 1 targeted web search sub-query targeting "web".
- For "hybrid" or "multi_hop": decompose into 2-3 focused sub-queries, each strictly labeled with source="internal", source="graph", or source="web".
- Make web search queries concise and keyword-dense (optimized for search engines like Google/Tavily). Never leak internal confidential keywords to web search queries.
- Keep "internal" sub-queries close to the user's own wording. Do not add qualifiers the
  user never said (never turn "who is Abinash" into "Abinash within the organization") —
  invented context pulls retrieval away from the passage that actually answers the question.

Output ONLY valid JSON matching this schema:
{
  "intent": "internal_rag" | "graph_rag" | "web_search" | "hybrid" | "direct_llm",
  "complexity": "simple" | "multi_hop",
  "reasoning": "<1 sentence rationale>",
  "sub_queries": [
    {
      "query": "<clean sub-query>",
      "source": "internal" | "graph" | "web",
      "reasoning": "<why this sub-query was assigned to this source>"
    }
  ]
}"""


async def query_analyzer_node(state: AgentState) -> AgentState:
    """Analyze query intent and produce decomposed, source-assigned sub-queries."""
    query = state.get("query", "").strip()
    route_history = list(state.get("route_history", []))
    route_history.append("query_analyzer")

    # Fast path: instant greeting detection
    if is_pleasantry(query):
        logger.info("Direct LLM fast-path triggered for greeting", query=query)
        return {
            **state,
            "intent": "direct_llm",
            "complexity": "simple",
            "is_direct_response": True,
            "sub_queries": [],
            "route_history": route_history,
        }

    try:
        from agents.corpus_manifest import get_corpus_documents
        from agents.llm_helper import generate_gemini_content

        # Routing without this is guesswork: the model is asked "internal or public?"
        # about a subject it has no way to look up. With the filenames in hand,
        # "who is abinash" against a corpus holding AbinashResume.pdf is unambiguous.
        corpus = state.get("corpus_documents")
        if corpus is None:
            corpus = await get_corpus_documents(state.get("tenant_id", ""))

        if corpus:
            corpus_block = (
                "The user has uploaded these documents. Assume any subject they cover is "
                "answerable internally:\n"
                + "\n".join(f"- {name}" for name in corpus[:40])
            )
        else:
            corpus_block = (
                "The user has no uploaded documents yet, so internal retrieval will return "
                "nothing. Use web_search or direct_llm as appropriate."
            )

        analysis_prompt = (
            f"{corpus_block}\n\nUser Query: {query}\n\nAnalyze this query and output JSON:"
        )

        raw_text = await generate_gemini_content(
            contents=analysis_prompt,
            system_instruction=QUERY_ANALYZER_SYSTEM_PROMPT,
            temperature=0.1,
            max_output_tokens=512,
            response_mime_type="application/json",
            # Classification, not generation quality — lite-first cuts latency and cost
            # without a quality tradeoff, and keeps this node off the model most likely
            # to be under heavy demand.
            candidate_models=["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.7-flash"],
        )

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(raw_text)

        intent = result.get("intent", "internal_rag")
        complexity = result.get("complexity", "simple")
        raw_sub_queries = result.get("sub_queries", [])

        sub_queries: list[SubQuery] = [
            {
                "query": sq.get("query", query),
                "source": sq.get("source", "internal"),
                "reasoning": sq.get("reasoning", ""),
            }
            for sq in raw_sub_queries
            if isinstance(sq, dict) and sq.get("query")
        ]

        # Ensure at least one sub-query exists if not direct_llm
        if not sub_queries and intent != "direct_llm":
            if intent == "web_search":
                default_source = "web"
            elif intent == "graph_rag":
                default_source = "graph"
            else:
                default_source = "internal"
            sub_queries = [{"query": query, "source": default_source, "reasoning": "Fallback full query"}]

        logger.info(
            "Query analysis complete",
            intent=intent,
            complexity=complexity,
            sub_queries_count=len(sub_queries),
            sub_queries=[f"[{sq['source']}] {sq['query']}" for sq in sub_queries],
        )

        return {
            **state,
            "intent": intent,
            "complexity": complexity,
            "is_direct_response": (intent == "direct_llm"),
            "sub_queries": sub_queries,
            "route_history": route_history,
        }

    except Exception as e:
        # Degrade toward the CHEAP route, not the expensive one. This handler used to
        # send everything to internal_rag, so a classifier outage turned every query —
        # including small talk the pleasantry check didn't catch — into a full
        # retrieve/synthesize/critique run. Only text that actually looks like a
        # question is worth paying for when we can no longer tell.
        fallback_is_question = looks_like_a_question(query)
        intent = "internal_rag" if fallback_is_question else "direct_llm"
        logger.warning(
            "Query analyzer LLM call failed; falling back on heuristic routing",
            error=str(e) or type(e).__name__,
            fallback_intent=intent,
        )
        return {
            **state,
            "intent": intent,
            "complexity": "simple",
            "is_direct_response": (intent == "direct_llm"),
            "sub_queries": (
                [{"query": query, "source": "internal", "reasoning": "Heuristic fallback"}]
                if fallback_is_question
                else []
            ),
            "route_history": route_history,
        }
