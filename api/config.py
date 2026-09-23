"""Application settings loaded from environment variables.

Uses pydantic-settings to read from .env files and environment. All config
flows through this single Settings object — no ad-hoc os.getenv() calls
scattered across the codebase.
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_JWT_SECRET = "change-me-to-a-random-secret-in-production"


class Settings(BaseSettings):
    """Central configuration. Reads from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # "development" | "staging" | "production". Anything other than development
    # turns on the startup safety checks in api/main.py.
    environment: str = "development"

    # Browser origins allowed to call this API, comma-separated. The frontend is
    # served from a different origin than the API, so a deployment whose real
    # domain is missing here has every request blocked by the browser.
    cors_allow_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000"
    )

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() not in ("development", "dev", "local", "test")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    # ── Postgres ────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://omnimind:omnimind@localhost:5432/omnimind"

    @field_validator("database_url")
    @classmethod
    def _force_async_driver(cls, value: str) -> str:
        """Normalize a managed provider's connection string to the async driver.

        Render, Heroku, Railway and friends hand out `postgresql://…` (and Heroku
        still emits the legacy `postgres://`). SQLAlchemy maps both to the DEFAULT
        driver — synchronous psycopg2 — and `create_async_engine` then rejects it at
        import time, which on a managed host surfaces as an unexplained boot loop
        rather than a readable error.

        Normalizing here rather than at the call site means every consumer benefits:
        the app engine, Alembic (which reads settings.database_url directly), and any
        script. Only the scheme is touched; credentials and query parameters such as
        `?sslmode=require` are left exactly as the provider supplied them.
        """
        for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
            if value.startswith(prefix):
                return value
        for legacy in ("postgresql://", "postgres://"):
            if value.startswith(legacy):
                return "postgresql+asyncpg://" + value[len(legacy):]
        return value

    # ── Redis ───────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Object storage ──────────────────────────────────────────
    # "s3" uses the MinIO/S3 settings below. "local" writes to storage_local_path
    # instead, which is what lets a single-node deployment run with no object-storage
    # service at all. "s3" additionally falls back to local on its own if no endpoint
    # answers at startup — see api/storage.py:ensure_bucket.
    storage_backend: str = "s3"
    storage_local_path: str = "./data/objects"

    # ── MinIO ───────────────────────────────────────────────────
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_name: str = "omnimind-documents"
    minio_secure: bool = False

    # ── JWT Auth ────────────────────────────────────────────────
    jwt_secret_key: str = "change-me-to-a-random-secret-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # When true, requests with a missing or invalid token are silently treated as the
    # built-in demo user instead of being rejected. That is a local-demo convenience
    # ONLY: it disables authentication for every protected route and collapses tenant
    # isolation, since tenant_id is derived from the resolved user. Defaults to off so
    # a deployment cannot inherit it by forgetting to set it.
    allow_demo_user_fallback: bool = False

    # ── Phase 2+ (optional — not required yet) ──────────────────
    pinecone_api_key: str | None = None
    pinecone_index_host: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "models/gemini-embedding-2"
    gemini_embedding_dim: int = 256
    gemini_thinking_budget: int = 0
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_base_url: str | None = None
    tavily_api_key: str | None = None

    # ── Voice mode (Gemini Live API — see docs/ADR/003-voice-mode.md) ──
    # Native audio speech-to-speech. Reuses gemini_api_key; a separate model
    # because the text flagship has no audio modality.
    voice_enabled: bool = True
    voice_live_model: str = "gemini-3.1-flash-live-preview"
    voice_name: str = "Puck"

    # Voice retrieval is deliberately tuned differently from text retrieval.
    # Query rewriting is off (it costs a full LLM round-trip and the Live model
    # already resolves coreferences in-session), and top_k is smaller because
    # spoken answers cite one or two sources, not five.
    voice_retrieval_top_k: int = 4
    voice_enable_rerank: bool = True

    # Evidence floor for the pre-speech gate. Voice cannot run the post-hoc
    # citation critic inline, so sufficiency is checked *before* generating
    # instead of faithfulness *after* — below this top score the tool reports
    # "no supporting evidence" and the model is instructed to say so.
    voice_evidence_floor: float = 0.35


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — parsed once, reused everywhere."""
    return Settings()
