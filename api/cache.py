"""Redis async client factory and FastAPI dependency.

Provides a single connection pool reused across the application lifetime.
"""

import redis.asyncio as aioredis

from api.config import get_settings

settings = get_settings()

redis_client = aioredis.from_url(
    settings.redis_url,
    decode_responses=True,
)


async def get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    """FastAPI dependency — returns the shared Redis client."""
    return redis_client  # type: ignore[return-value]


async def redis_ping() -> bool:
    """Health-check helper — returns True if Redis responds to PING."""
    try:
        return await redis_client.ping()  # type: ignore[return-value]
    except Exception:
        return False
