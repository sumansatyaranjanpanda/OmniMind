# ADR 002: Model Gateway, Frontend Stack, and Episodic Memory

**Status:** Accepted  
**Date:** 2026-08-30  
**Context:** During Phase 5–6 implementation, three decisions from ADR 001 were revised or expanded without follow-up documentation. This ADR ratifies those changes and explains the reasoning.

---

## Decision 1: Gemini Direct (Not OpenRouter Gateway)

### Chosen
Use Google's Gemini SDK directly (`google-genai`). Call the Gemini API from `agents/llm_helper.py` without an intermediary gateway.

**Primary model:** `gemini-3.7-flash`  
**Fallbacks:** `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`

### Alternatives Considered
1. **Keep OpenRouter gateway** (original ADR 001 plan)
   - Provider-agnostic routing
   - Single API key management
   - Easy swaps between providers

2. **Use OpenAI API directly**
   - More mature than Gemini
   - Larger community

### Why Gemini Direct Won
- **Fast iteration:** Gemini's API is stable and feature-rich. Direct calls eliminate gateway abstraction overhead during active development.
- **Model availability:** `gemini-3.7-flash` supports inline multimodal reasoning (images/tables/video) required for Phase 4. Vision processing would need custom glue code behind OpenRouter.
- **Cost per token:** Gemini pricing is competitive; faster inference reduces per-query spend.
- **Thinking mode support:** `gemini-thinking-budget` parameter (future) only works with Gemini API, not via intermediary gateway.

### Consequences
- **Vendor lock-in:** Switching to a different LLM provider later requires refactoring `agents/llm_helper.py` and all node implementations.
- **Cost model tied to Gemini:** If Gemini rates spike, no fallback to cheaper provider without rewrite.
- **Mitigation:** Keep `openrouter_api_key` and `openai_api_key` in config as optional; a future ADR can propose a thin abstraction layer if multi-provider support becomes a requirement.

---

## Decision 2: React + Vite (Not Next.js)

### Chosen
Frontend: React 18 + Vite + TypeScript. Client-side rendering (CSR), no server-side rendering.

**Config:** `frontend/vite.config.ts`  
**Dev server:** `npm run dev` → Vite on port 5173

### Alternatives Considered
1. **Keep Next.js** (original ADR 001 plan)
   - SSR for SEO and initial page load performance
   - Built-in API routes
   - Streaming SSE support (partially)

2. **Use Svelte/SvelteKit**
   - Compiler-based optimization
   - Less boilerplate

### Why React + Vite Won
- **Dev speed:** Vite's HMR (hot module replacement) is significantly faster than Next.js cold starts, critical during active iteration.
- **Scope fit:** OmniMind is an internal enterprise tool, not public-facing. SEO and SSR are not required.
- **Bundle size:** Vite + React (CSR) produces a smaller initial bundle than Next.js + SSR runtime.
- **Simplicity:** No server-side logic in frontend. All business logic lives in FastAPI. Frontend just sends queries and renders responses.
- **Streaming:** SSE streaming (POST `/chat/stream`) works identically in CSR via `EventSource` API.

### Consequences
- **No SSR:** Initial page load time depends on JS bundle evaluation. For internal tools, this is acceptable.
- **No built-in API routes:** All backend calls explicitly target FastAPI (`api/main.py`). Separation is cleaner but requires more CORS setup.
- **Mitigation:** If public-facing SEO becomes a requirement, switch to Next.js later. Current CSR architecture is not a blocker.

---

## Decision 3: Pinecone Multi-Purpose (Documents + Episodic Memory)

### Chosen
Use a **single Pinecone serverless index** with **multiple namespaces**:
- `{tenant_id}` namespaces for document chunks (Phase 2)
- `memory-{user_id}` namespace for episodic conversation memory (Phase 6.7)

**Hybrid retrieval:** Dense (Pinecone) + Sparse (BM25 via `rank-bm25` library) merged via Reciprocal Rank Fusion (k=60).

### Alternatives Considered
1. **Use Pinecone's built-in hybrid** (original ADR 001 plan)
   - Single index with dense + sparse vectors in one query
   - Simpler API

2. **Separate vector DBs**
   - Pinecone for dense
   - Elasticsearch for sparse
   - Clean separation of concerns

3. **Use pgvector in PostgreSQL**
   - Everything in Postgres (no new service)
   - Simpler ops for small scale

### Why Multi-Purpose Pinecone Won

#### Dense + Sparse Hybrid
- **Flexibility:** Rank-bm25 (local Python library) gives fine-grained control over sparse weighting. Pinecone's hybrid would be opaque.
- **Cost:** No extra third-party service. BM25 is CPU-only, not cloud-dependent.
- **RRF fusion (k=60):** Proven in production RAG systems. Outperforms learning-to-rank on small result sets.
- **Fallback:** If Pinecone is down, dense search fails gracefully; BM25 still works locally.

#### Episodic Memory Namespace
- **Single service:** Simplifies ops and secrets management. One Pinecone account, one index, multiple namespaces.
- **Namespace isolation:** `memory-{user_id}` prevents cross-user leakage. Query filters exclude other threads, so episodic context doesn't pollute document retrieval.
- **Semantic recall:** Episodic memory is inherently dense (semantic similarity). Reusing Pinecone avoids standing up a second vector DB.
- **Non-blocking:** Background embedding (post-response) means zero latency impact on chat.

### Consequences
- **Hybrid deviation:** ADR 001 specified "hybrid dense+sparse in one index"; we're doing separate dense (Pinecone) + sparse (BM25) merged at application level.
  - **Consequence:** Two retrieval calls per query (one dense, one sparse).
  - **Mitigation:** Both are fast (< 100ms combined). RRF fusion is deterministic and reproducible.
- **Namespace coupling:** Document chunks and episodic memories share one Pinecone account and index.
  - **Consequence:** Namespace collisions if naming is not careful. Quota sharing.
  - **Mitigation:** Strict naming convention (`memory-{user_id}` never conflicts with `{tenant_id}`). Monitor quota separately.
- **Cost scaling:** If episodic memory grows (many users, many threads), Pinecone storage/query costs rise linearly.
  - **Mitigation:** Archive old episodic memories after 90 days; or move to a cheaper tier if that becomes a problem.

---

## Going Forward

1. **These decisions are locked in.** If OmniMind needs to become multi-provider or public-facing later, that's a new ADR (003+).
2. **Fallback configs remain optional** in `api/config.py`: `openrouter_api_key`, `openai_api_key` for future flexibility.
3. **Keep single-index design** as long as Pinecone cost remains < 5% of budget. If it exceeds that, revisit.
4. **Monitor Gemini rate limits.** If they become a bottleneck, propose ADR 003 for multi-provider routing.

---

## Ratified By
- Architecture owner (you)
- Implementation team (Claude Code)
- Date: 2026-08-30
