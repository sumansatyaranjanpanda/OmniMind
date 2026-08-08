"""Health-check router.

GET /health returns service status and connectivity checks for Postgres,
Redis, and MinIO. No auth required — this is used by Docker health checks
and monitoring.
"""

import structlog
from fastapi import APIRouter
from sqlalchemy import text

from api.cache import redis_ping
from api.database import async_session_factory
from api.storage import minio_healthy

logger = structlog.get_logger()
router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, object]:
    """Return overall health plus per-service connectivity status."""
    pg_ok = False
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
            pg_ok = True
    except Exception as exc:
        logger.warning("postgres_health_check_failed", error=str(exc))

    redis_ok = await redis_ping()
    minio_ok = await minio_healthy()

    all_ok = pg_ok and redis_ok and minio_ok
    return {
        "status": "ok" if all_ok else "degraded",
        "services": {
            "postgres": "up" if pg_ok else "down",
            "redis": "up" if redis_ok else "down",
            "minio": "up" if minio_ok else "down",
        },
    }
