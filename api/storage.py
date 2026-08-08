"""MinIO (S3-compatible) object storage client and FastAPI dependency.

Provides bucket auto-creation at startup and a reusable client instance.
"""

from miniopy_async import Minio  # type: ignore[import-untyped]

from api.config import get_settings

settings = get_settings()

minio_client = Minio(
    endpoint=settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=settings.minio_secure,
)


async def ensure_bucket() -> None:
    """Create the configured bucket if it doesn't already exist."""
    exists = await minio_client.bucket_exists(settings.minio_bucket_name)
    if not exists:
        await minio_client.make_bucket(settings.minio_bucket_name)


async def get_storage() -> Minio:
    """FastAPI dependency — returns the shared MinIO client."""
    return minio_client


async def minio_healthy() -> bool:
    """Health-check helper — returns True if the bucket is reachable."""
    try:
        await minio_client.bucket_exists(settings.minio_bucket_name)
        return True
    except Exception:
        return False
