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


async def upload_file(object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Uploads bytes to MinIO and returns the object name (key)."""
    import io
    
    stream = io.BytesIO(data)
    length = len(data)
    
    await minio_client.put_object(
        settings.minio_bucket_name,
        object_name,
        stream,
        length,
        content_type=content_type,
    )
    return object_name


async def download_file(object_name: str) -> bytes:
    """Downloads a file from MinIO and returns the bytes."""
    response = await minio_client.get_object(
        settings.minio_bucket_name,
        object_name,
    )
    try:
        return await response.read()
    finally:
        if hasattr(response, "close"):
            response.close()
        if hasattr(response, "release"):
            response.release()


async def delete_file(object_name: str) -> None:
    """Deletes an object from MinIO. Not an error if it's already gone."""
    await minio_client.remove_object(settings.minio_bucket_name, object_name)
