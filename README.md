# OmniMind

Enterprise RAG/agent platform: ingest multi-format documents, retrieve with hybrid
search, reason over the evidence with a multi-agent LangGraph pipeline, and return
answers where every claim carries a citation back to the page it came from.

## What it does

- **Ingest** PDF / DOCX / PPTX / XLSX / CSV / MD / HTML with structure preserved —
  headings, tables and figures survive parsing instead of being flattened to text.
- **Retrieve** with hybrid search: dense vectors (Pinecone) and sparse BM25 fused by
  reciprocal rank fusion, then reranked by a cross-encoder.
- **Reason** over the result with a LangGraph agent graph — query analysis, retrieval,
  optional graph/web engines, synthesis, and a citation critic that blocks any answer
  scoring below 0.85 on faithfulness.
- **Answer** with inline `[^N]` citations, or say "insufficient evidence". Answers
  sourced from the web because the corpus had nothing say so explicitly.
- **Talk** — a speech-to-speech voice mode over the Gemini Live API, with a tiered
  tool router so a spoken turn doesn't pay for the full text pipeline.

## Architecture

```
POST /chat  (client sends query + thread_id — the server owns history)
  → input guardrail (injection/jailbreak block, PII mask)
  → semantic cache (Redis, cosine ≥ 0.90)
  → context rewriter (coreference) ∥ episodic memory recall
  → query analysis ∥ document retrieval        ← run concurrently
  → evidence gate: documents above the floor → synthesize; nothing → web, disclosed
  → source fusion → synthesizer (inline citations)
  → citation critic (faithfulness ≥ 0.85) → rewrite loop on failure, max 2
  → output guardrail (secret scrub) → response
```

Voice takes a separate lane (`WS /voice/stream`) with three tiers — conversational,
fast document search, and full critic-gated research — because the text graph's
sequential LLM round-trips are exactly what makes a spoken turn unusable.

Full detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and the ADRs under
[docs/ADR/](docs/ADR/).

## Performance

Latency work is driven by live measurement, not estimates. On one representative
document question, end to end: **68s → 17-22s → 4.6s to first token.**

What actually moved the needle:

| Change | Why it mattered |
|---|---|
| Adaptive model routing | Provider latency moved ~10x within an hour and the configured primary model was answering 0 of 5 calls. Candidate order is now derived from measured per-model latency (EWMA), because two hand-written orderings in this repo were both stale within a day. |
| Hedged requests | A stalled model no longer blocks the chain — the next candidate starts in parallel after a hedge delay and the first usable response wins. Latency becomes `hedge_delay + fastest live model` instead of the sum of preceding timeouts. |
| Concurrent analysis + retrieval | Retrieval never needed the classifier's output. Once documents have answered, the analyzer gets a bounded grace window, then the request proceeds on the documents-only route. |
| Memoized query embeddings | The same query string was being embedded three times per request at ~0.8s each. |
| Per-model thinking config | Only some models accept `thinking_budget=0`; applying it blindly failed and applying it nowhere was slow. Also fixed a chain-of-thought leak where a model emitted its internal reasoning as the visible answer. |

## Local development

```bash
cp .env.example .env          # fill in real values
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

docker compose -f infrastructure/docker-compose.yml up -d   # postgres, redis, minio
alembic -c infrastructure/alembic.ini upgrade head
uvicorn api.main:app --reload --port 8000

cd frontend && npm install && npm run dev                   # http://localhost:5173
```

```bash
pytest tests/unit -k "not (auth or health)" -v   # 256 tests, no live services needed
pytest tests/unit -v                             # full run, needs the compose stack
ruff check . && ruff format . && mypy .
```

## Deployment

`Dockerfile` at the repo root builds a single container that serves the API and the
built React frontend from one origin and one port, with Postgres running alongside:

```bash
docker build -t omnimind:prod .
docker run -p 7860:7860 --env-file .env omnimind:prod
```

Single-container is an MVP trade, not an architectural claim. It fits a free
single-container host and removes CORS and cross-service networking from the list of
things a demo can fail on. What it gives up: **the database is ephemeral** — it lives
in the container filesystem, so accounts, conversations and document metadata reset
when the container restarts. Vectors persist, because Pinecone is external.

Set `DATABASE_URL` to a managed Postgres and the entrypoint skips the in-container
database entirely — that is the upgrade path to a real deployment.

## Configuration

See [.env.example](.env.example). The Phase 1 core four (Postgres, Redis, MinIO, JWT)
are required; everything else is optional and degrades rather than failing:
Redis down → in-memory cache, Pinecone unset → local vector store, Cohere unset →
local FlashRank reranking, Tavily unset → DuckDuckGo.

In production (`ENVIRONMENT` set to anything but development) the app refuses to start
with the default JWT signing key or with the demo-user fallback enabled.
