# OmniMind

Enterprise RAG/agent platform: ingest multi-format documents, retrieve with hybrid search, reason over evidence with a multi-agent pipeline, and return citation-backed answers.

## Quickstart

```bash
# 1. Clone and set up environment
cp .env.example .env          # fill in real values
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -e ".[dev]"

# 2. Start infrastructure
docker compose -f infrastructure/docker-compose.yml up -d

# 3. Run database migrations
alembic -c infrastructure/alembic.ini upgrade head

# 4. Start the API
uvicorn api.main:app --reload

# 5. Verify
curl http://localhost:8000/health
```

## Development

```bash
# Lint & format
ruff check . && ruff format .

# Type check
mypy .

# Tests (requires Postgres running via Docker Compose)
pytest tests/unit -v
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system overview.

## Project Status

Currently in **Phase 1 (Foundation)**. See `claude.md` for the phase roadmap.
