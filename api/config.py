"""Application settings loaded from environment variables.

Uses pydantic-settings to read from .env files and environment. All config
flows through this single Settings object — no ad-hoc os.getenv() calls
scattered across the codebase.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. Reads from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Postgres ────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://omnimind:omnimind@localhost:5432/omnimind"

    # ── Redis ───────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

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

    # ── Phase 2+ (optional — not required yet) ──────────────────
    pinecone_api_key: str | None = None
    pinecone_index_host: str | None = None
    openrouter_api_key: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — parsed once, reused everywhere."""
    return Settings()
