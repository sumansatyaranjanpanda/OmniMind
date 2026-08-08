# OmniMind Architecture

## Overview

OmniMind is an enterprise RAG (Retrieval-Augmented Generation) platform that ingests
multi-format documents, retrieves relevant content with hybrid search, reasons over
evidence with a multi-agent LangGraph pipeline, and returns citation-backed answers.

## System Architecture (Phase 1)

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                    │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐   │
│  │  Health   │  │   Auth   │  │   Future Routers    │   │
│  │  Router   │  │  Router  │  │   (Phase 2+)        │   │
│  └────┬─────┘  └────┬─────┘  └─────────────────────┘   │
│       │              │                                   │
│  ┌────┴──────────────┴──────────────────────────────┐   │
│  │              Dependencies (deps.py)               │   │
│  │    get_db() | get_current_user() | get_redis()    │   │
│  └────┬──────────────┬──────────────┬───────────────┘   │
│       │              │              │                    │
└───────┼──────────────┼──────────────┼────────────────────┘
        │              │              │
   ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐
   │Postgres │   │  Security  │  │  Redis  │
   │(SQLAlch │   │ (JWT+pwd)  │  │ (cache) │
   │ +Alembic│   └────────────┘  └─────────┘
   └─────────┘
        │
   ┌────▼────┐
   │  MinIO  │
   │ (files) │
   └─────────┘
```

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| API | FastAPI + Pydantic v2 | Async-first, auto-generated OpenAPI docs, Pydantic for typed contracts |
| ORM | SQLAlchemy 2.0 (async) | Mature, typed, great Alembic integration |
| Migrations | Alembic | Standard for SQLAlchemy, supports async |
| Database | PostgreSQL 16 | Source of truth, ACID, mature |
| Cache/Queue | Redis 7 | Fast K/V store, future pub/sub for workers |
| Object Storage | MinIO | S3-compatible, local dev, swap for S3 in prod |
| Auth | JWT (HS256) + bcrypt | Stateless auth, industry-standard password hashing |
| Logging | structlog | Structured JSON logging, easy to query |
| Vector DB | Pinecone (Phase 2+) | Managed serverless, hybrid dense+sparse search |
| LLM Access | OpenRouter (Phase 2+) | Provider-agnostic gateway |
| Orchestration | LangGraph (Phase 5+) | Supervisor + subgraph agent pattern |

## Module Responsibilities

See [CODEBASE_WALKTHROUGH.md](CODEBASE_WALKTHROUGH.md) for detailed per-module documentation.

## Security Principles

1. **No hardcoded secrets** — all config via env vars
2. **LLM-generated SQL is read-only** — no DROP/DELETE/UPDATE/ALTER/TRUNCATE
3. **Document content is untrusted** — never let it alter system prompts or trigger tools
4. **All answers cite evidence** — or explicitly state "insufficient evidence"
5. **No silent failures** — no `except: pass`
