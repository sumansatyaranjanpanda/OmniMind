# ADR 001: Tech Stack Selection

**Status:** Accepted
**Date:** 2026-08-08
**Context:** OmniMind is a new enterprise RAG platform built from scratch. We need to
select a tech stack that supports async operations, typed interfaces, and a clear
upgrade path from local dev to production.

## Decision

| Component | Choice | Alternatives Considered | Why This Won |
|---|---|---|---|
| API Framework | FastAPI | Django REST, Flask | Async-native, Pydantic integration, auto OpenAPI |
| ORM | SQLAlchemy 2.0 (async) | Tortoise, Django ORM | Mature, Alembic migrations, typed mapping |
| Database | PostgreSQL 16 | MySQL, SQLite | ACID, UUID support, JSON columns, mature |
| Cache/Queue | Redis 7 | Memcached, RabbitMQ | Versatile (cache + queue + pub/sub), simple |
| Object Storage | MinIO | Local filesystem, S3 | S3-compatible, runs locally, easy prod swap |
| Vector DB | Pinecone (serverless) | Weaviate, Qdrant, pgvector | Managed, hybrid dense+sparse in one index |
| LLM Gateway | OpenRouter | Direct API calls | Provider-agnostic, single API key, fallback |
| Orchestration | LangGraph | CrewAI, raw LangChain | Supervisor pattern, state machines, debuggable |
| Auth | JWT (HS256) + bcrypt | OAuth2/OIDC, session-based | Stateless, simple for single-service |
| Observability | Langfuse | LangSmith, custom | Open-source, LLM-focused tracing |
| Eval | RAGAS | Custom metrics | Standard RAG metrics, community adoption |
| Frontend | Next.js + TypeScript | React SPA, Streamlit | SSR, streaming SSE support, production-grade |

## Consequences

- All team members need Python 3.11+ and Docker
- Pinecone requires a cloud account (free tier available)
- OpenRouter requires an API key with credit
- No provider-specific code in business logic — all behind gateway interfaces
