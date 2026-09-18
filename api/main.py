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
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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

# ── Frontend (single-origin deployment) ─────────────────────────
# When a built frontend is present, this process serves it from the same origin as
# the API. That is what makes a one-container deployment possible, and it sidesteps
# CORS entirely — the browser only ever talks to one host, so cors_allow_origins
# stops being a thing a deployment can get wrong and be mysteriously broken by.
#
# Absent in local development, where Vite serves the frontend on :5173 and proxies
# to this API, so the block below simply doesn't engage.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# Paths owned by the API. A request under one of these that reached the catch-all
# matched no route, and must 404 as JSON rather than being handed the SPA shell —
# a client calling a mistyped endpoint should see an error, not 200 OK and HTML.
_API_PREFIXES = (
    "health", "auth", "documents", "search", "chat",
    "conversations", "graph", "voice", "docs", "redoc", "openapi.json",
)

if _FRONTEND_DIST.is_dir():
    _assets = _FRONTEND_DIST / "assets"
    if _assets.is_dir():
        app.mount("/assets", StaticFiles(directory=_assets), name="assets")

    # response_model=None because the return annotation is a union of two Response
    # types. FastAPI otherwise tries to build a Pydantic response model from it and
    # refuses at import time with "Invalid args for response field", which takes the
    # whole process down at startup rather than failing on a request.
    @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
    async def serve_frontend(full_path: str) -> FileResponse | JSONResponse:
        """Serve built assets, falling back to index.html so client routing works.

        Registered last on purpose: Starlette matches routes in registration order,
        so every API route above still wins. Only genuinely unmatched paths land here.
        """
        if full_path.split("/", 1)[0] in _API_PREFIXES:
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        candidate = (_FRONTEND_DIST / full_path).resolve()
        # Containment check, not a convenience: without it a crafted path like
        # ../../etc/passwd would escape the dist directory and serve arbitrary
        # files off the container filesystem.
        if (
            full_path
            and candidate.is_file()
            and candidate.is_relative_to(_FRONTEND_DIST.resolve())
        ):
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")

    logger.info("frontend_static_serving_enabled", path=str(_FRONTEND_DIST))
