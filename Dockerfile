# OmniMind — single-container production image.
#
# Serves the API and the built React frontend from ONE origin and ONE port, with
# Postgres running alongside in the same container. That is a deliberate MVP
# trade, not an architectural claim: it fits a free single-container host, and it
# removes CORS and cross-service networking from the list of things a demo can
# fail on. See README for what this gives up (chiefly: the database is ephemeral).
#
# Local development is unaffected and still uses infrastructure/docker-compose.yml
# with Postgres, Redis and MinIO as separate services.

# ── Stage 1: build the frontend ────────────────────────────────
FROM node:20-slim AS frontend

WORKDIR /build
# Copy manifests alone first so `npm ci` is cached against dependency changes
# rather than re-running on every source edit.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ── Stage 2: python dependencies ───────────────────────────────
FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# torch arrives transitively via docling/sentence-transformers/timm, used for local
# FlashRank reranking and Docling parsing — never GPU inference, since model access
# is Gemini-direct per ADR 002. Left to default resolution pip pulls the CUDA build
# and drags in nvidia-cublas/cudnn/cuda-toolkit/triton: several gigabytes that took
# builds past 20 minutes for zero benefit on a machine with no GPU. Installing the
# CPU wheel first means pip sees torch already satisfied when it later resolves
# docling's `torch<3.0.0,>=2.2.2`, so the CUDA variant is never considered.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY api/ ./api/
COPY security/ ./security/
COPY ingestion/ ./ingestion/
COPY parsing/ ./parsing/
COPY chunking/ ./chunking/
COPY retrieval/ ./retrieval/
COPY reranking/ ./reranking/
COPY agents/ ./agents/
COPY core/ ./core/
COPY model_gateway/ ./model_gateway/
COPY evaluation/ ./evaluation/
COPY observability/ ./observability/
COPY workers/ ./workers/
RUN pip install --no-cache-dir .


# ── Stage 3: runtime ───────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PGDATA=/home/appuser/pgdata

# postgresql-client supplies pg_isready, which start.sh waits on rather than
# sleeping a fixed number of seconds and hoping.
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Document upload writes the original file to storage before ingestion, so storage is
# on the critical path, not optional. MinIO was going to run in-container to provide
# it, but their binary distribution now returns HTTP 410 Gone, so instead the app's
# filesystem storage backend handles it (api/storage.py). Fewer moving parts and less
# memory besides; set STORAGE_BACKEND=s3 with real credentials for durable storage.
ENV STORAGE_BACKEND=local \
    STORAGE_LOCAL_PATH=/home/appuser/objects

# uid 1000 because that is the user the host runs the container as, and because
# Postgres refuses to start as root at all.
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY api/ ./api/
COPY security/ ./security/
COPY ingestion/ ./ingestion/
COPY parsing/ ./parsing/
COPY chunking/ ./chunking/
COPY retrieval/ ./retrieval/
COPY reranking/ ./reranking/
COPY agents/ ./agents/
COPY core/ ./core/
COPY model_gateway/ ./model_gateway/
COPY evaluation/ ./evaluation/
COPY observability/ ./observability/
COPY workers/ ./workers/
COPY infrastructure/alembic.ini ./infrastructure/alembic.ini
COPY infrastructure/migrations/ ./infrastructure/migrations/

# Built frontend — api/main.py mounts this directory when it exists and serves the
# SPA from the same origin as the API.
COPY --from=frontend /build/dist ./frontend/dist

COPY scripts/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh \
    && mkdir -p /home/appuser/pgdata /home/appuser/objects /home/appuser/.cache \
    && chown -R appuser:appuser /home/appuser /app

USER appuser

# Model weights download here on first use; without HOME set to a writable path
# the download fails as a permission error at startup.
ENV HOME=/home/appuser

EXPOSE 7860

CMD ["/usr/local/bin/start.sh"]
