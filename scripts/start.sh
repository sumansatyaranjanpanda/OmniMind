#!/usr/bin/env bash
# Container entrypoint: bring up Postgres, migrate, then serve.
#
# `set -e` deliberately NOT used. The API is built to boot with degraded
# dependencies — api/main.py logs a warning and continues when Postgres, Redis or
# object storage is unreachable — so a failure here should still leave a process
# running that can report its own health, rather than exiting and leaving the host
# to report nothing but "container crashed".
set -uo pipefail

PORT="${PORT:-7860}"
PGBIN="$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | head -1)"

log() { echo "[start.sh] $*"; }

# DATABASE_URL being already set means a real managed Postgres was supplied, which
# is strictly better than the in-container one: use it and skip all of this.
if [ -n "${PGBIN}" ] && [ -z "${DATABASE_URL:-}" ]; then
    export PATH="${PGBIN}:${PATH}"

    if [ ! -s "${PGDATA}/PG_VERSION" ]; then
        log "initialising postgres cluster at ${PGDATA}"
        initdb -D "${PGDATA}" -U omnimind --auth=trust >/dev/null 2>&1 \
            && log "initdb complete" \
            || log "initdb FAILED — continuing; the API will report degraded health"
    fi

    # Exported before Postgres is even known to be up, because the application reads
    # this at import time and an EMPTY value is not "no database" to SQLAlchemy — it
    # is an unparseable URL, which raises during module import and kills the process
    # outright. A well-formed URL pointing at a database that happens to be down is
    # handled gracefully by api/main.py; an empty one defeats that entirely.
    export DATABASE_URL="postgresql+asyncpg://omnimind@127.0.0.1:5432/omnimind"

    # unix_socket_directories must be writable by this user. Postgres refuses to run
    # as root, and the packaged default (/var/run/postgresql) is root-owned, so
    # leaving it unset fails with "could not create lock file ... Permission denied"
    # even though the TCP listener below is what we actually connect over.
    #
    # listen_addresses is loopback-only: reachable by this container's own processes
    # and nothing else, even when the host publishes the app port. --auth=trust above
    # is only safe because of that, so the two must stay together.
    log "starting postgres on 127.0.0.1:5432"
    pg_ctl -D "${PGDATA}" \
        -o "-c listen_addresses=127.0.0.1 -p 5432 -c unix_socket_directories=/tmp" \
        -l /tmp/pg.log start >/dev/null 2>&1

    for _ in $(seq 1 30); do
        pg_isready -h 127.0.0.1 -p 5432 -U omnimind >/dev/null 2>&1 && break
        sleep 1
    done

    if pg_isready -h 127.0.0.1 -p 5432 -U omnimind >/dev/null 2>&1; then
        log "postgres is ready"

        psql -h 127.0.0.1 -U omnimind -d postgres -tc \
            "SELECT 1 FROM pg_database WHERE datname='omnimind'" 2>/dev/null | grep -q 1 \
            || createdb -h 127.0.0.1 -U omnimind omnimind 2>/dev/null

        log "running migrations"
        (cd /app && alembic -c infrastructure/alembic.ini upgrade head 2>&1 | tail -5) \
            || log "migrations FAILED — continuing"
    else
        log "postgres did not become ready; starting API with a degraded database"
        tail -20 /tmp/pg.log 2>/dev/null
    fi
fi

log "storage backend: ${STORAGE_BACKEND:-s3} (${STORAGE_LOCAL_PATH:-unset})"

log "starting uvicorn on 0.0.0.0:${PORT}"
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT}" --workers 1
