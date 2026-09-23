#!/usr/bin/env bash
# Entrypoint for the slim image: migrate against the managed database, then serve.
#
# No `set -e`: api/main.py is built to boot with degraded dependencies and report
# that through /health. Exiting here instead would replace a diagnosable "degraded"
# response with an opaque container crash loop.
set -uo pipefail

# Free hosts assign the port at runtime and expect the process to honour it.
PORT="${PORT:-8000}"

log() { echo "[start-slim] $*"; }

if [ -n "${DATABASE_URL:-}" ]; then
    log "running migrations"
    (cd /app && alembic -c infrastructure/alembic.ini upgrade head 2>&1 | tail -5) \
        || log "migrations FAILED — continuing so /health can report the cause"
else
    log "DATABASE_URL is not set — starting without migrations"
fi

log "starting uvicorn on 0.0.0.0:${PORT}"
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
