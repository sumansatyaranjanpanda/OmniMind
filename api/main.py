"""OmniMind API — FastAPI application entry point.

Creates the app, wires up routers, and runs startup checks:
- Verify Postgres connectivity
- Verify Redis connectivity
- Ensure MinIO bucket exists

Run with: uvicorn api.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from sqlalchemy import text

from api.cache import redis_client, redis_ping
from api.database import async_engine, async_session_factory
from api.routers import auth, health
from api.storage import ensure_bucket

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook."""
    # ── Startup ─────────────────────────────────────────────
    logger.info("startup_begin")

    # Postgres
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        logger.info("postgres_connected")
    except Exception as exc:
        logger.error("postgres_connection_failed", error=str(exc))
        raise

    # Redis
    if await redis_ping():
        logger.info("redis_connected")
    else:
        logger.warning("redis_connection_failed")

    # MinIO — create bucket if missing
    try:
        await ensure_bucket()
        logger.info("minio_bucket_ready")
    except Exception as exc:
        logger.warning("minio_connection_failed", error=str(exc))

    logger.info("startup_complete")
    yield

    # ── Shutdown ────────────────────────────────────────────
    logger.info("shutdown_begin")
    await redis_client.aclose()  # type: ignore[attr-defined]
    await async_engine.dispose()
    logger.info("shutdown_complete")


app = FastAPI(
    title="OmniMind",
    description="Enterprise RAG/agent platform with hybrid search and citation-backed answers.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Routers ─────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(auth.router)
