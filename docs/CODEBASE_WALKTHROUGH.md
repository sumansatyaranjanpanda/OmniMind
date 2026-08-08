# OmniMind — Codebase Walkthrough (Phase 1)

> Last updated: Phase 1 (Foundation)
> This document covers the entire codebase as it stands. It's written so someone who
> hasn't read the code can explain the system in an interview.

---

## Module: `api/`

### What it does
The API module is the entry point for the entire application. It hosts a FastAPI
application with two router groups (health and auth), dependency injection for database
sessions and user authentication, and startup/shutdown lifecycle hooks that verify
connectivity to Postgres, Redis, and MinIO.

### Why it's built this way
FastAPI was chosen over Flask/Django because it's async-first (matching our async
SQLAlchemy and async Redis/MinIO clients), generates OpenAPI docs automatically, and
has first-class support for Pydantic v2 models. The alternative — Django REST Framework —
was rejected because its sync-first nature would require wrkarounds for our async stack.

### How it connects
- **Inbound**: HTTP requests from clients (or Docker health checks)
- **Outbound**: Calls `api/database.py` for DB sessions, `api/cache.py` for Redis,
  `api/storage.py` for MinIO, `security/` for JWT and password operations
- **Data in**: JSON request bodies (Pydantic schemas)
- **Data out**: JSON responses (Pydantic schemas)

### Key files
| File | Purpose |
|---|---|
| `api/main.py` | App creation, lifespan hooks, router wiring |
| `api/config.py` | Centralized settings via pydantic-settings |
| `api/database.py` | Async SQLAlchemy engine, session factory, `get_db()` |
| `api/cache.py` | Redis async client, `get_redis()`, `redis_ping()` |
| `api/storage.py` | MinIO client, `ensure_bucket()`, `minio_healthy()` |
| `api/deps.py` | `get_current_user()` dependency (JWT → User lookup) |
| `api/routers/health.py` | `GET /health` with per-service status |
| `api/routers/auth.py` | `POST /auth/signup`, `POST /auth/login`, `GET /auth/me` |
| `api/schemas/auth.py` | Request/response Pydantic models for auth |
| `api/models/user.py` | SQLAlchemy `User` ORM model |

### How to verify it
```bash
# Start infrastructure
docker compose -f infrastructure/docker-compose.yml up -d postgres redis minio

# Run the API
uvicorn api.main:app --reload

# Health check
curl http://localhost:8000/health

# Signup → Login → Access protected route
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"pass123"}'

curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"pass123"}'

# Use the token from login response:
curl http://localhost:8000/auth/me -H "Authorization: Bearer <token>"
```

---

## Module: `security/`

### What it does
Contains two focused utilities: password hashing (bcrypt via passlib) and JWT
creation/decoding (HS256 via python-jose). These are the only places in the codebase
that touch cryptographic operations.

### Why it's built this way
Separated from `api/` so that security logic is independently testable and reusable
by future modules (e.g., workers that need to validate tokens). bcrypt was chosen over
argon2 for broader ecosystem support; HS256 over RS256 because we're a single-service
system (no need for public key verification by third parties).

### How it connects
- **Called by**: `api/routers/auth.py` (hash at signup, verify at login, create token),
  `api/deps.py` (decode token for auth middleware)
- **Depends on**: `api/config.py` for JWT secret/algorithm/expiry settings

### Key files
| File | Purpose |
|---|---|
| `security/password.py` | `hash_password()`, `verify_password()` |
| `security/jwt.py` | `create_access_token()`, `decode_access_token()` |

### How to verify it
```bash
pytest tests/unit/test_password.py tests/unit/test_jwt.py -v
```

---

## Module: `infrastructure/`

### What it does
Houses all deployment and infrastructure configuration: Docker Compose for local dev,
the API Dockerfile, Alembic config, and database migrations.

### Why it's built this way
Docker Compose brings up the full local stack in one command. Alembic is the standard
migration tool for SQLAlchemy — the alternative (raw SQL scripts) was rejected because
autogenerate + version tracking is essential for a growing schema. Pinecone is excluded
from Docker Compose because it's a managed cloud service.

### How it connects
- `docker-compose.yml` starts Postgres, Redis, MinIO, and the API container
- Alembic reads `api/config.py` for the database URL and `api/database.py` for the
  ORM metadata to generate migrations

### Key files
| File | Purpose |
|---|---|
| `infrastructure/docker-compose.yml` | Postgres 16, Redis 7, MinIO, API service |
| `infrastructure/Dockerfile` | Multi-stage Python 3.11 build |
| `infrastructure/alembic.ini` | Alembic configuration |
| `infrastructure/migrations/env.py` | Async migration runner |
| `infrastructure/migrations/versions/0001_*.py` | Create `users` table |

### How to verify it
```bash
# Start everything
docker compose -f infrastructure/docker-compose.yml up -d

# Run migrations
alembic -c infrastructure/alembic.ini upgrade head

# Verify tables exist
docker exec omnimind-postgres psql -U omnimind -c "\dt"
```

---

## Module: `tests/`

### What it does
Unit tests using pytest + pytest-asyncio. Tests run against a real Postgres instance
(from Docker Compose) with per-test transaction rollback for isolation.

### Why it's built this way
Using real Postgres (not SQLite) catches dialect-specific issues early — UUID columns,
`server_default=func.now()`, etc. all work correctly because we test against the same
engine we deploy with. Each test wraps its work in a transaction that rolls back, so
tests are fast and isolated without needing to drop/recreate the schema.

### How it connects
- `tests/conftest.py` creates a test engine, overrides the `get_db` dependency, and
  provides an `AsyncClient` fixture for endpoint testing
- Individual test files cover password, JWT, health, and auth endpoints

### How to verify it
```bash
# Ensure Postgres is running
docker compose -f infrastructure/docker-compose.yml up -d postgres

# Run tests
pytest tests/unit -v
```

---

## Skeleton Modules (not yet active)

The following modules exist as empty packages with docstrings describing their future
purpose. They are **not built yet** — creating them now ensures the monorepo structure
matches the spec from day one.

| Module | Phase | Future purpose |
|---|---|---|
| `ingestion/` | 2 | Document loaders (PDF, DOCX, etc.) |
| `parsing/` | 2 | Structure-aware parsing |
| `chunking/` | 2 | Chunking strategies + router |
| `retrieval/` | 2–3 | Embeddings + Pinecone hybrid search |
| `reranking/` | 3 | Cross-encoder reranking |
| `agents/` | 5 | LangGraph supervisor + subgraphs |
| `model_gateway/` | 2+ | Provider-agnostic LLM calls |
| `evaluation/` | 2+ | RAGAS eval harness |
| `observability/` | 2+ | Langfuse tracing |
| `workers/` | 2+ | Background job consumers |
| `frontend/` | TBD | Next.js app |
