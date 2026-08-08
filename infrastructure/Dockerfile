# OmniMind API — multi-stage Docker build
# Stage 1: install dependencies
# Stage 2: copy source and run

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Stage 1: Dependencies ──────────────────────────────────────
FROM base AS deps

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ── Stage 2: Application ───────────────────────────────────────
FROM base AS runtime

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source
COPY api/ ./api/
COPY security/ ./security/
COPY ingestion/ ./ingestion/
COPY parsing/ ./parsing/
COPY chunking/ ./chunking/
COPY retrieval/ ./retrieval/
COPY reranking/ ./reranking/
COPY agents/ ./agents/
COPY model_gateway/ ./model_gateway/
COPY evaluation/ ./evaluation/
COPY observability/ ./observability/
COPY workers/ ./workers/
COPY infrastructure/alembic.ini ./infrastructure/alembic.ini
COPY infrastructure/migrations/ ./infrastructure/migrations/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
