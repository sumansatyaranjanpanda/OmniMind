# CLAUDE.md — OmniMind

## What this is

OmniMind is an enterprise RAG/agent platform: ingest multi-format documents, retrieve
with hybrid search, reason over evidence with a multi-agent LangGraph pipeline, and
return citation-backed answers. Built as a portfolio-grade system, not a demo script —
but scoped so it actually ships.

**This is a brand-new, standalone project, starting from zero.** It is not a fork,
extension, or continuation of NEXUS, Bhasha, or any other prior project — no code,
modules, data, prompts, or infrastructure are carried over. Every module listed below
gets written from scratch inside this repo. If something looks similar to a past
project, that's convergent design, not reused code — verify, don't assume.

**This file is operational, not aspirational.** If a decision isn't made here, don't
invent scaffolding for it — ask or add an ADR. Full long-term vision lives in
`docs/VISION.md`; this file only describes what's actually being built right now.

## Current phase: Phase 1 (Foundation) — nothing built yet

Do not start Phase 3+ work (multimodal, agents, knowledge graph, K8s, multi-tenancy)
until the phases before it pass their Definition of Done below. Scope creep into later
phases is the single biggest risk on a solo-built project — resist it, including when
it "would only take an hour."

| Phase | Contents | Status |
|---|---|---|
| 1 | Monorepo, FastAPI, Postgres, Redis, MinIO, auth, Docker Compose, CI | ⏳ next |
| 2 | Ingestion, structure-aware parsing, chunking, embeddings, vector search, citations | 🧊 not started |
| 3 | Hybrid search (BM25 + dense), reranking, query rewriting | 🧊 not started |
| 4 | Multimodal (images/tables/OCR) | 🧊 not started |
| 5 | Agents (planner, SQL agent, web agent) | 🧊 not started |
| 6+ | Knowledge graph, multi-tenancy, K8s, evaluation platform, cost/latency opt | 🧊 not started |

## Tech stack (decided — do not swap without an ADR)

- **API**: FastAPI + Pydantic v2 + async SQLAlchemy 2.0
- **Orchestration**: LangGraph (supervisor + subgraphs per agent)
- **DB**: PostgreSQL (source of truth), Alembic for migrations
- **Cache/queue**: Redis
- **Object storage**: MinIO (S3-compatible)
- **Vector DB**: Pinecone (serverless index), hybrid search via dense + sparse (BM25) vectors in the same index — no local container, it's a managed cloud service
- **Eval**: RAGAS (faithfulness, context precision/recall)
- **Observability/tracing**: Langfuse
- **Async events** (Phase 6+ only, not before): Kafka
- **Model access**: OpenRouter as gateway; no provider hard-coded into business logic
- **Frontend**: Next.js + TypeScript + Tailwind, streaming via SSE
- **Containers**: Docker Compose for dev; Kubernetes deferred to Phase 9

## Repo structure

```
api/            FastAPI app, routers, deps, auth
ingestion/      Loaders per file type -> normalized Document objects
parsing/        Structure-aware parsers (headings/tables/figures preserved)
chunking/       Chunking strategies + router (recursive/semantic/structural)
retrieval/      Embedding calls, Pinecone client, hybrid search, fusion
reranking/      Cross-encoder rerank stage
agents/         LangGraph graphs: supervisor + per-agent subgraphs
model_gateway/  Provider-agnostic model calls, routing, fallback
evaluation/     RAGAS harness, eval datasets, regression tracking
security/       Auth, RBAC, tenant scoping helpers
observability/  Langfuse wiring, structured logging, metrics
workers/        Background job consumers (embedding, parsing)
infrastructure/ docker-compose.yml, Dockerfiles, migrations
frontend/       Next.js app
docs/           README, ARCHITECTURE.md, SECURITY.md, ADR/, VISION.md
tests/          unit/, integration/, e2e/
```

## Core request flow (Phase 2–3 scope)

```
Request → Auth → Query classification → Retrieval (hybrid) →
Rerank → Context assembly → Generation → Citation check → Response
```

Full agentic flow (planner, SQL/web/graph agents, critic) is Phase 5+. Don't build
agent scaffolding for phases we're not in — a single retrieval-and-generate path is
correct for now.

## Environment / secrets

Pinecone is a managed cloud service, not a local container — `docker-compose` only
runs Postgres, Redis, and MinIO. Set in `.env` (never commit):

```
PINECONE_API_KEY=
PINECONE_INDEX_HOST=        # from Pinecone console after index creation
OPENROUTER_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

## Development commands

```bash
# start local stack (postgres, redis, minio — not pinecone)
docker compose -f infrastructure/docker-compose.yml up -d

# api
cd api && uvicorn main:app --reload

# migrations
alembic upgrade head
alembic revision --autogenerate -m "description"

# tests
pytest tests/unit -v
pytest tests/integration -v
pytest tests/e2e -v

# lint / format
ruff check . && ruff format .
mypy .

# eval suite
python -m evaluation.run_eval --dataset evaluation/datasets/v1.jsonl
```

(Update these once actual scripts exist — don't leave stale commands here.)

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
- No giant catch-all functions; no silent `except: pass`
- Tests land with the code that needs them, not after
- Don't claim a feature works without running it

## Definition of Done — Phase 1

- [ ] Monorepo structure created matching the layout below (empty modules with `__init__.py` is fine — structure exists)
- [ ] FastAPI app boots with a health-check endpoint
- [ ] Postgres reachable via SQLAlchemy async engine, first Alembic migration runs
- [ ] Redis reachable from the API
- [ ] MinIO reachable, one bucket created programmatically
- [ ] Auth: signup/login issuing JWT, at least one protected route
- [ ] `docker-compose.yml` brings up Postgres + Redis + MinIO + API together
- [ ] CI pipeline runs lint + unit tests on push
- [ ] `docs/CODEBASE_WALKTHROUGH.md` created (see mandatory section below) even though it'll be short at this stage

## Definition of Done — Phase 2

- [ ] PDF/DOCX/PPTX/XLSX/CSV/MD/HTML ingestion working end to end
- [ ] Structure preserved (headings/tables/figures), not flattened to plain text
- [ ] Chunking router picks a strategy per document type
- [ ] Embeddings stored in Pinecone with metadata filters (document_id, tenant stub via namespace)
- [ ] A query returns chunks with correct provenance and a working citation back to source
- [ ] Unit + integration tests for ingestion → retrieval path
- [ ] One RAGAS eval run producing a faithfulness/context-precision baseline
- [ ] Eval/test documents are freshly created for this project — no reuse of NEXUS/Bhasha data

## Phase-end codebase walkthrough (mandatory)

Before moving to the next phase, Claude Code must produce a full walkthrough of the
codebase **as it stands so far** — not just a diff of what changed this phase. Write
it to `docs/CODEBASE_WALKTHROUGH.md`, overwriting/updating the previous version, and
also summarize it directly in the chat response.

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

## Claude Code working rules

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