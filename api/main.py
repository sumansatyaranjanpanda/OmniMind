"""OmniMind API — FastAPI application entry point.

Creates the app, wires up routers, and runs startup checks:
- Verify Postgres connectivity
- Verify Redis connectivity
- Ensure MinIO bucket exists

Run with: uvicorn api.main:app --reload
"""

import asyncio
import os

# Disable TorchDynamo / TorchInductor compilation on Windows (avoids missing cl.exe MSVC compiler error)
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from api.cache import redis_client, redis_ping
from api.config import DEFAULT_JWT_SECRET, get_settings
from api.database import async_engine, async_session_factory
from api.routers import auth, chat, chat_stream, conversations, documents, graph, health, search, voice
from api.storage import ensure_bucket
from reranking.reranker import warm_flashrank_fallback
from retrieval.pinecone_client import close_async_index

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook."""
    # ── Startup ─────────────────────────────────────────────
    logger.info("startup_begin", environment=settings.environment)

    # Refuse to serve production traffic with the signing key that ships in the
    # repo. Anyone holding it can mint a valid token for any email address and
    # read that account's documents — and a default that "works" is exactly the
    # kind of thing that survives a deploy unnoticed. Fail loudly instead.
    if settings.is_production:
        if settings.jwt_secret_key == DEFAULT_JWT_SECRET:
            raise RuntimeError(
                "JWT_SECRET_KEY is still the built-in default. Generate one with:\n"
                "  python -c \"import secrets; print(secrets.token_urlsafe(48))\"\n"
                "and set it in the environment before starting in production."
            )
        if settings.allow_demo_user_fallback:
            raise RuntimeError(
                "ALLOW_DEMO_USER_FALLBACK is enabled outside development. That "
                "serves unauthenticated requests as the demo user and disables "
                "tenant isolation. Set it to false."
            )
    elif settings.jwt_secret_key == DEFAULT_JWT_SECRET:
        logger.warning(
            "Using the default JWT signing key — fine locally, never in production"
        )

    # Postgres
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
        logger.info("postgres_connected")
    except Exception as exc:
        logger.warning("postgres_connection_failed", error=str(exc))

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

    # Force the FlashRank fallback model to download/load now rather than during a live
    # request. Confirmed under load testing: on a fresh checkout the first Cohere failure
    # blocks a real user's request on a model download instead of getting the fast local
    # fallback it was supposed to get. Runs in a thread since Ranker() is synchronous I/O.
    try:
        await asyncio.to_thread(warm_flashrank_fallback)
        logger.info("flashrank_fallback_warmed")
    except Exception as exc:
        logger.warning("flashrank_warmup_failed", error=str(exc))

    logger.info("startup_complete")
    yield

    # ── Shutdown ────────────────────────────────────────────
    logger.info("shutdown_begin")
    await redis_client.aclose()  # type: ignore[attr-defined]
    await async_engine.dispose()
    await close_async_index()
    logger.info("shutdown_complete")


app = FastAPI(
    title="OmniMind",
    description="Enterprise RAG/agent platform with hybrid search and citation-backed answers.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS Middleware ─────────────────────────────────────────────
# Origins come from config, not a hard-coded localhost list — a deployment served
# from its real domain would otherwise have every browser request blocked.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(chat_stream.router)
app.include_router(conversations.router)
app.include_router(graph.router)
app.include_router(voice.router)
