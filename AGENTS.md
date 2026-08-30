# AGENTS.md — OmniMind

**This is the single canonical instructions file for this repo.** `CLAUDE.md` (Claude Code)
and `GEMINI.md` (Antigravity/Gemini CLI) both import this file — don't fork content into
them. If you're an agent and this content reached you via an import, you're in the right
place; if you opened `CLAUDE.md`/`GEMINI.md` directly and see a raw `@AGENTS.md` line instead
of this content, your tool didn't resolve the import — read this file directly instead.

## What this is

OmniMind is an enterprise RAG/agent platform: ingest multi-format documents, retrieve
with hybrid search, reason over evidence with a multi-agent LangGraph pipeline, and
return citation-backed answers. Built as a portfolio-grade system, not a demo script —
but scoped so it actually ships.

**This is a brand-new, standalone project, starting from zero.** It is not a fork,
extension, or continuation of NEXUS, Bhasha, or any other prior project — no code,
modules, data, prompts, or infrastructure are carried over. If something looks similar to
a past project, that's convergent design, not reused code — verify, don't assume.

**This file is operational, not aspirational.** If a decision isn't made here, don't
invent scaffolding for it — ask or add an ADR. Full long-term vision lives in
`docs/VISION.md`; this file only describes what's actually being built right now.

## Current phase: code is de facto through Phase ~6.7, but the process that's supposed to gate that is broken — see "Process gaps" below before trusting any phase label

**2026-08-28 reconciliation note:** a second agent (Gemini CLI, working through the old
GEMINI.md) built out most of Phases 2–7 directly in the working tree without going through
this file's own rules — no incremental ADRs for stack changes, no `CODEBASE_WALKTHROUGH.md`
updates past Phase 4, no committed history (still one commit: "Phase 1 Foundation
completion"). Any status claim of "Complete & Verified" found elsewhere is self-reported —
the table below is what a code+test inspection actually supports. Treat gaps as open work,
not as "basically done."

| Phase | Contents | Status (verified 2026-08-28) |
|---|---|---|
| 1 | Monorepo, FastAPI, Postgres, Redis, MinIO, auth, Docker Compose, CI | ✅ code + CI present; DB-dependent tests (`test_auth`, `test_health`) only run against a live `docker compose up` stack, not verified in this pass |
| 2 | Ingestion, structure-aware parsing, chunking, embeddings, vector search, citations | ⚠️ built (Docling + chunking router + Pinecone client exist) but **no unit tests for ingestion/parsing/documents router**, and no captured RAGAS baseline output despite the harness existing |
| 3 | Hybrid search (BM25 + dense), reranking, query rewriting | ✅ implemented, unit-tested (`test_hybrid_search.py`, `test_reranker.py`, `test_query_rewriter.py` pass) |
| 4 | Multimodal (images/tables/OCR) | ✅ Docling + Gemini vision captioning implemented (`parsing/parsers.py`); no dedicated test coverage |
| 5 | Agents (LangGraph CRAG-style graph: analyzer, retriever, web search, fusion, synthesizer, critic) | ✅ implemented and unit-tested (`test_agents.py`, 98 passing across the mocked suite) |
| 6 | Knowledge graph, evaluation platform, multi-tenancy/RBAC, SSE streaming | ✅ implemented and unit-tested (`test_graph_rag.py`, `test_evaluation.py`, `test_rbac.py`); RBAC is per-request scoping, not full multi-tenant isolation — verify before calling it that |
| 6.5–6.7 | Guardrails (injection/PII/secret-scrub), multi-turn memory (window + rolling summary + cross-thread episodic recall) | ✅ **2026-08-30**: memory rearchitected — Alembic migration written (`b71a4f2e9c3d`), server is now the sole authority for history (client sends only `query`+`thread_id`), summary folds incrementally (O(1)/turn, not the original full-recompute bug), episodic recall added via a Pinecone `memory-{user_id}` namespace. Unit-tested (`test_guardrails.py`, `test_context_memory.py`, 15/15 passing). **Verified same day** — `docker compose up -d && alembic upgrade head` applied `e620348fc329 -> b71a4f2e9c3d` cleanly against a live Postgres; `\d conversations`/`\d messages` confirmed all columns (incl. `message_count`/`summarized_through_count`) and FKs match the models. Full suite re-run against the live stack: 106/106 passing |
| 7 | Frontend (React + Vite, not the Next.js from ADR 001) | ⚠️ built, not covered by the last reconciliation pass (no frontend test run performed) |
| 8+ | Kubernetes, multi-agent swarms | 🧊 not started |

Do not mark any phase "done" in the table above, or start Phase 8 work, until the gaps in
this table are closed and `docs/CODEBASE_WALKTHROUGH.md` is brought current (it currently
stops at Phase 4). See "Process gaps to close" near the end of this file.

## Tech stack (ADR 001 decision vs. what's actually in the code — see note)

`docs/ADR/001-tech-stack.md` (2026-08-08) is still the only ADR in the repo. Several rows
below were superseded in the working tree without a follow-up ADR, which breaks the
non-negotiable rule "never introduce a new tech-stack component without proposing an ADR
first." Until ADR 002 is written and accepted, treat the "as-built" column as unsanctioned
fact, not as a green light to keep building on it further.

| Component | ADR 001 decision | As-built in code (verified) | Needs ADR 002? |
|---|---|---|---|
| API | FastAPI + Pydantic v2 + async SQLAlchemy 2.0 | matches | no |
| DB | PostgreSQL, Alembic migrations | matches — Conversation/Message migration (`b71a4f2e9c3d`) written 2026-08-30, verified against a live DB same day | no |
| Cache/queue | Redis | matches; also used as a semantic-response cache (cosine ≥ 0.90, TTL 1h) | maybe (scope expansion) |
| Object storage | MinIO | matches | no |
| Vector DB | Pinecone serverless, hybrid dense+sparse in one index | `retrieval/pinecone_client.py` exists with local in-memory fallback; hybrid fusion implemented as separate dense (Pinecone) + sparse (`rank-bm25`) with reciprocal rank fusion (k=60), not single-index hybrid. **2026-08-30**: also now used for a second purpose — `retrieval/memory_store.py` upserts/queries a per-user `memory-{user_id}` namespace for episodic conversation recall, isolated from the document-chunk namespaces | yes — both the original hybrid-index deviation and the new memory-namespace use |
| Orchestration | LangGraph, supervisor + subgraphs | matches (`agents/graph.py`, `agents/nodes/`) | no |
| Model access | **OpenRouter gateway**, no provider hard-coded | `model_gateway/` is an empty stub — never implemented. `agents/llm_helper.py`, `retrieval/pinecone_client.py`, `agents/nodes/{evaluator,query_rewriter}.py`, and `parsing/parsers.py` all call the Gemini SDK (`google-genai`) directly | **yes — this is the biggest deviation** |
| Reranking | (not specified in ADR 001) | Cohere Rerank v3.5 (`rerank-v3.5`) with local FlashRank fallback | yes |
| Web search | (not in original scope until Phase 5) | Tavily AI with DuckDuckGo fallback | yes |
| Guardrails | (not in original scope) | native heuristic module (`core/guardrails.py`) — injection/jailbreak blocking, PII masking, secret scrubbing | yes |
| Eval | RAGAS (faithfulness, context precision/recall) | harness + dataset exist (`evaluation/`), no baseline run captured yet | no |
| Observability/tracing | Langfuse | matches | no |
| Frontend | **Next.js** + TypeScript + Tailwind, SSR, SSE streaming | **React 18 + Vite + TypeScript** (no SSR), SSE streaming via `/chat/stream` | **yes** |
| Async events (Phase 6+) | Kafka | not present; not needed yet at this scale | no (still correctly deferred) |
| Containers | Docker Compose for dev; K8s deferred | matches | no |

**Active model/service config** (from `api/config.py`, for when you need the exact strings):
primary LLM `gemini-3.7-flash`; lighter-weight nodes (query analyzer, context rewriter) use
`gemini-3.5-flash-lite`; embeddings `models/gemini-embedding-2` (Matryoshka, 256-dim);
reranker `rerank-v3.5`; citation critic faithfulness gate ≥ 0.85 with a max-2 query-rewrite
retry loop on failure.

**Action before any further stack-touching work**: write ADR 002 covering the Gemini-direct
model access decision and the Next.js→React/Vite frontend swap at minimum — those two
silently reversed explicit ADR 001 decisions, not just added to them.

## Repo structure (as-built, verified 2026-08-28)

```
api/            FastAPI app, routers (health/auth/documents/search/chat/chat_stream/
                 conversations/graph), deps, models (user/document/conversation),
                 schemas, storage, config
ingestion/      pipeline.py — loaders per file type -> normalized Document objects
parsing/        parsers.py — Docling structure-aware parsing + Gemini vision captioning
chunking/       strategies.py — chunking strategies + router (recursive/semantic/structural)
retrieval/      pinecone_client, hybrid_search, query_rewriter, graph_store, pipeline,
                 models, memory_store.py (episodic cross-thread recall, Pinecone
                 `memory-{user_id}` namespace)
reranking/      reranker.py — Cohere Rerank v3.5 with local FlashRank fallback
agents/         graph.py (LangGraph StateGraph), state.py, llm_helper.py (direct Gemini calls),
                 memory/service.py (sliding window + incremental rolling-summary fold,
                 O(1)/turn — see Phase 6.5–6.7 row above),
                 nodes/ (cache_check, context_rewriter, query_analyzer, retriever,
                 graph_retriever, web_search, source_fusion, synthesizer, critic,
                 query_rewriter, guardrails, direct_llm, evaluator), tools/
core/           guardrails.py — injection/jailbreak, PII masking, secret scrubbing
model_gateway/  STUB ONLY — __init__.py describes an OpenRouter gateway that was never built;
                 see "Tech stack" table above
evaluation/     dataset_generator.py, metrics.py, run_eval.py, datasets/v1.jsonl (RAGAS)
security/       rbac.py — role/permission scoping helpers
observability/  langfuse_client.py, tracer.py, metrics.py
infrastructure/ docker-compose.yml, Dockerfile, migrations/versions/ (Alembic — users,
                 documents, conversations/messages as of `b71a4f2e9c3d` — see gaps below)
frontend/       React 18 + Vite + TypeScript app (not Next.js — see "Tech stack" above)
docs/           ARCHITECTURE.md, VISION.md, ADR/001-tech-stack.md, CODEBASE_WALKTHROUGH.md
                 (stale — stops at Phase 4)
tests/unit/     test_agents, test_auth, test_chunking, test_context_memory, test_evaluation,
                 test_graph_rag, test_guardrails, test_health, test_hybrid_search, test_jwt,
                 test_password, test_query_rewriter, test_rbac, test_reranker,
                 test_retrieval_pipeline — no ingestion/parsing/documents-router tests yet
```

`workers/` from the original plan doesn't exist — no background job consumers were built;
everything currently runs inline in the request path. Flag if that becomes a latency problem.

## Core request flow (as-built — the agentic flow originally scoped for Phase 5+ is already live)

```
POST /chat  (client sends only query + optional thread_id — server owns history)
  → Load conversation context (Postgres: sliding window + rolling summary)
  → Input guardrail (injection/jailbreak block, PII mask)
  → Semantic cache check (Redis, cosine ≥ 0.90) — hit short-circuits to cached answer
  → Context rewriter (coreference resolution) + episodic memory recall (Pinecone,
      cross-thread), run concurrently
  → Query analyzer (intent: direct_llm / internal_rag / web_search / hybrid)
  → [Retriever (dense+BM25+RRF+rerank)] and/or [Web search (Tavily → DDG fallback)]
      run in parallel when hybrid
  → Source fusion (unify + provenance numbering)
  → Synthesizer (inline citations; window + summary + episodic memory as context)
  → Citation critic (faithfulness ≥ 0.85) → on fail, query rewriter loop (max 2 retries)
  → Output guardrail (secret/credential scrub)
  → Response (+ cache write)
```

This flow is real (see `agents/graph.py` + `agents/nodes/`, unit-tested in `test_agents.py`),
but it was built without the phase-by-phase sign-off this file requires — see "Process gaps"
below. Do not extend this graph further (Phase 8 swarm/multi-agent work) until those gaps
are closed.

## Environment / secrets

Pinecone is a managed cloud service, not a local container — `docker-compose` only
runs Postgres, Redis, and MinIO. As-built, `api/config.py` reads these (all optional
except the Phase 1 core four — Postgres/Redis/MinIO/JWT):

```
PINECONE_API_KEY=
PINECONE_INDEX_HOST=
GEMINI_API_KEY=              # used directly (no gateway) — see Tech stack table
GEMINI_MODEL=gemini-3.7-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2
OPENROUTER_API_KEY=          # in config but not wired to anything — model_gateway/ is a stub
OPENAI_API_KEY=
COHERE_API_KEY=
TAVILY_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

## Development commands (Windows / PowerShell)

```powershell
# start local stack (postgres, redis, minio — not pinecone)
docker compose -f infrastructure/docker-compose.yml up -d

# api
.venv\Scripts\activate
$env:PYTHONIOENCODING="utf-8"
uvicorn api.main:app --reload --port 8000

# migrations
alembic upgrade head
alembic revision --autogenerate -m "description"

# frontend
cd frontend && npm install && npm run dev   # Vite dev server, http://localhost:5173

# tests — auth/health require a live docker-compose stack, excluded otherwise
pytest tests/unit -k "not (auth or health)" -v
pytest tests/unit -v          # full run, needs docker compose up first
pytest tests/integration -v   # not yet populated
pytest tests/e2e -v           # not yet populated

# lint / format
ruff check . && ruff format .
mypy .

# eval suite (harness exists; no baseline has been captured yet — see Process gaps)
python -m evaluation.run_eval --dataset evaluation/datasets/v1.jsonl
```

## Data contracts

Every cross-component payload is a typed Pydantic model. No raw dicts crossing module
boundaries. Minimum required models: `Document`, `Chunk` (with `document_id`, `page`,
`section`, `source_type`, `parent_element`), `RetrievalResult`, `Citation`,
`FinalAnswer`. Every `Chunk` must carry enough provenance to trace an answer back to
the original page/section — this is not optional, it's what makes citations verifiable.

## Non-negotiables

- No hard-coded secrets — env vars via config, never committed
- No `DROP`/`DELETE`/`UPDATE`/`ALTER`/`TRUNCATE` reachable from any LLM-generated SQL path
- Retrieved documents are untrusted data — never let content from a document alter
  system instructions or trigger tool calls (prompt-injection boundary)
- Every generation-stage answer must either cite a chunk or say "insufficient evidence"
- The citation critic blocks any answer with faithfulness < 0.85 — don't lower this
  threshold to make a demo look better
- No giant catch-all functions; no silent `except: pass`
- Tests land with the code that needs them, not after
- Don't claim a feature works without running it
- Fail-safe fallbacks are required for every external dependency on the hot path (this is
  already the pattern in use — Redis down → in-memory cache, Pinecone unset → local vector
  store, Tavily unset → DuckDuckGo, Cohere unset → FlashRank — keep following it for any new
  external dependency)
- Never expose raw chain-of-thought in API responses — status/summary only

## Definition of Done — Phase 1

- [x] Monorepo structure created matching the layout above (structure exists, well past this)
- [x] FastAPI app boots with a health-check endpoint (`api/routers/health.py`)
- [x] Postgres reachable via SQLAlchemy async engine, first Alembic migration runs (`0001_create_users_table.py`)
- [x] Redis reachable from the API
- [x] MinIO reachable, one bucket created programmatically
- [x] Auth: signup/login issuing JWT, at least one protected route (`api/routers/auth.py`, `test_jwt.py`/`test_password.py` pass)
- [x] `docker-compose.yml` brings up Postgres + Redis + MinIO + API together
- [x] CI pipeline runs lint + unit tests on push (`.github/workflows/ci.yml`)
- [x] `docs/CODEBASE_WALKTHROUGH.md` created — **but it was never updated past Phase 4; needs a fresh full pass, see Process gaps**

Note: `test_auth.py`/`test_health.py` weren't re-run against a live `docker compose up`
stack in the last reconciliation pass — checked based on code + CI evidence, not a fresh
live run. Verify with `docker compose up -d` + full `pytest tests/unit -v` before trusting
this blindly.

## Definition of Done — Phase 2

- [x] PDF/DOCX/PPTX/XLSX/CSV/MD/HTML ingestion working end to end (Docling-based, `parsing/parsers.py`)
- [x] Structure preserved (headings/tables/figures), not flattened to plain text
- [x] Chunking router picks a strategy per document type (`chunking/strategies.py`)
- [ ] Embeddings stored in Pinecone with metadata filters — Pinecone client exists with a local
      in-memory fallback; not confirmed against a real Pinecone index in this pass
- [x] A query returns chunks with correct provenance and a working citation back to source
- [ ] Unit + integration tests for ingestion → retrieval path — **retrieval is tested
      (`test_hybrid_search.py`, `test_retrieval_pipeline.py`); ingestion/parsing/documents
      router has no unit tests at all**
- [ ] One RAGAS eval run producing a faithfulness/context-precision baseline — harness and
      `evaluation/datasets/v1.jsonl` exist, but **no baseline output has been captured anywhere in the repo**
- [x] Eval/test documents are freshly created for this project

## Phase-end codebase walkthrough (mandatory)

Before moving to the next phase, the agent must produce a full walkthrough of the codebase
**as it stands so far** — not just a diff of what changed this phase. Write it to
`docs/CODEBASE_WALKTHROUGH.md`, overwriting/updating the previous version, and also
summarize it directly in the chat response.

For every module that currently exists, cover:

- **What it does** — plain-language description, no jargon dump
- **Why it's built this way** — the alternative that was considered and why this
  option won, in a sentence or two
- **How it connects** — what calls it, what it calls, what data goes in/out
- **How to verify it** — the command to run to see it working

Write it so that someone who has not read the code can explain the whole system to an
interviewer afterward. This is not optional documentation debt — it's the actual
deliverable that makes the project useful to you, and it should get easier to write
each phase, not harder, since it's cumulative.

## Process gaps to close (found during 2026-08-28 reconciliation; updated 2026-08-30)

**Resolved 2026-08-30 (two gaps):**
1. Conversation/Message migration (`b71a4f2e9c3d`) verified against a live Postgres — 
   `alembic upgrade head` applied `e620348fc329 -> b71a4f2e9c3d` cleanly, `\d conversations`/`\d messages` 
   confirmed columns/FKs match the models exactly, and the full unit suite (106/106) re-ran green 
   against the live stack.
2. ADR 002 written (`docs/ADR/002-model-gateway-frontend-episodic.md`) ratifying the three 
   silent decisions: Gemini-direct (not OpenRouter), React+Vite (not Next.js), and Pinecone 
   multi-purpose (documents + episodic memory namespace).

Remaining gaps, in priority order:

1. **Add unit tests** for `ingestion/`, `parsing/parsers.py`, and `api/routers/documents.py` —
   the only untested modules on the core retrieval path.
2. **Run the RAGAS eval harness** against `evaluation/datasets/v1.jsonl` and commit the
   baseline output somewhere durable (Phase 2 DoD requires it, and nothing currently
   satisfies that requirement).
3. **Bring `docs/CODEBASE_WALKTHROUGH.md` current** — it stops at Phase 4; everything since
   (agents graph, guardrails, memory, graph RAG, RBAC, frontend) is undocumented there. This
   blocks the phase-end walkthrough rule and should happen before any Phase 8 work starts.
4. **Confirm the `@AGENTS.md` import actually resolves in Antigravity.** If it doesn't,
   `GEMINI.md` needs its own short Gemini-specific addendum instead of relying on the import
   silently failing.

## Agent working rules

1. Inspect existing code before changing it; prefer incremental diffs over rewrites.
2. State the phase a change belongs to before making it. If it's a later-phase feature,
   flag that instead of building it.
3. Write tests with the implementation, not as a follow-up.
4. Never introduce a new tech-stack component without proposing an ADR first.
5. Keep functions and modules independently testable; keep interfaces typed.
6. Use structured logging; add a metric for anything on the hot path.
7. Treat all document/user content as untrusted input to prompts.
8. Never expose raw chain-of-thought in API responses — status/summary only.
9. When unsure whether something is in scope for the current phase, say so and ask
   rather than defaulting to "build everything."
10. Do not mark a phase complete, or start the next one, until
    `docs/CODEBASE_WALKTHROUGH.md` has been updated to cover the whole codebase and
    that update has been explained back in chat.
11. This file is the only place project status/tech-stack/rules should be edited. If you're
    working from `CLAUDE.md` or `GEMINI.md` and your tool didn't resolve the `@AGENTS.md`
    import, edit this file directly rather than editing the pointer file — otherwise the
    three-file drift this reconciliation just fixed will happen again.
