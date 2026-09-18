"""Object storage: S3/MinIO, with a local filesystem backend as the fallback.

Every other external dependency on the hot path already degrades instead of failing
— Redis down falls back to an in-memory cache, Pinecone unset to a local vector
store, Cohere unset to local FlashRank, Tavily unset to DuckDuckGo. Object storage
was the one exception, and it is not an optional one: `documents.py` writes the
original file here before ingestion runs, so an unreachable bucket failed uploads at
the first step with a connection error.

The filesystem backend closes that gap. It engages when STORAGE_BACKEND=local, and
also automatically when no S3 endpoint is reachable at startup, so a single-node or
single-container deployment needs no object-storage service at all. Durability is
whatever the underlying disk gives you — which is the honest trade, and the reason
S3 stays the default rather than becoming a second-class path.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

import structlog

from api.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Resolved at startup by ensure_bucket(). Kept as module state rather than recomputed
# per call so a single probe decides the backend for the process's lifetime — a
# per-request check would add a network round trip to every upload.
_use_local: bool | None = None


def _local_root() -> Path:
    return Path(settings.storage_local_path).expanduser().resolve()


def _local_path(object_name: str) -> Path:
    """Resolve an object key under the storage root, refusing escapes.

    Keys reach this from request-derived values, so a key like
    `../../etc/passwd` must not resolve outside the root. Checked after
    resolution rather than by inspecting the string, since `..` can hide behind
    symlinks and encodings.
    """
    root = _local_root()
    candidate = (root / object_name).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"object key escapes the storage root: {object_name!r}")
    return candidate


def _minio_client() -> Any:
    from miniopy_async import Minio  # type: ignore[import-untyped]

    return Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


# Built lazily so importing this module never requires the S3 client library, which
# is what lets a local-backend deployment omit it from its dependencies entirely.
_client: Any = None


def _client_or_none() -> Any:
    global _client
    if _use_local:
        return None
    if _client is None:
        try:
            _client = _minio_client()
        except Exception as exc:
            logger.warning("Could not construct the S3 client", error=str(exc))
            return None
    return _client


async def ensure_bucket() -> None:
    """Pick a backend and make it ready. Called once, at startup.

    Prefers S3 when it answers. Falls back to the filesystem rather than raising,
    because a failure here would otherwise mean no uploads work at all.
    """
    global _use_local

    if settings.storage_backend.strip().lower() == "local":
        _use_local = True
        _local_root().mkdir(parents=True, exist_ok=True)
        logger.info("storage_backend_local", path=str(_local_root()))
        return

    try:
        client = _minio_client()
        if not await client.bucket_exists(settings.minio_bucket_name):
            await client.make_bucket(settings.minio_bucket_name)
        _use_local = False
        globals()["_client"] = client
        logger.info("storage_backend_s3", endpoint=settings.minio_endpoint)
        return
    except Exception as exc:
        _use_local = True
        _local_root().mkdir(parents=True, exist_ok=True)
        logger.warning(
            "Object storage unreachable; falling back to local filesystem storage. "
            "Uploads will work, but files live on this node's disk only.",
            error=str(exc),
            path=str(_local_root()),
        )


async def get_storage() -> Any:
    """FastAPI dependency — the shared S3 client, or None on the local backend."""
    return _client_or_none()


async def storage_healthy() -> bool:
    """True when the active backend can be reached."""
    if _use_local:
        try:
            return _local_root().is_dir()
        except Exception:
            return False
    client = _client_or_none()
    if client is None:
        return False
    try:
        await client.bucket_exists(settings.minio_bucket_name)
        return True
    except Exception:
        return False


# Kept under the original name so existing callers and health checks keep working.
minio_healthy = storage_healthy


async def upload_file(
    object_name: str, data: bytes, content_type: str = "application/octet-stream"
) -> str:
    """Store bytes under `object_name` and return the key."""
    if _use_local:
        target = _local_path(object_name)
        # Blocking file IO moved off the event loop: uploads can be tens of MB, and
        # writing those inline would stall every other in-flight request.
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)
        return object_name

    client = _client_or_none()
    if client is None:
        raise RuntimeError("No storage backend is available for upload")

    await client.put_object(
        settings.minio_bucket_name,
        object_name,
        io.BytesIO(data),
        len(data),
        content_type=content_type,
    )
    return object_name


async def download_file(object_name: str) -> bytes:
    """Read an object's bytes."""
    if _use_local:
        return await asyncio.to_thread(_local_path(object_name).read_bytes)

    client = _client_or_none()
    if client is None:
        raise RuntimeError("No storage backend is available for download")

    response = await client.get_object(settings.minio_bucket_name, object_name)
    try:
        return await response.read()
    finally:
        if hasattr(response, "close"):
            response.close()
        if hasattr(response, "release"):
            response.release()


async def delete_file(object_name: str) -> None:
    """Remove an object. Already being gone is not an error."""
    if _use_local:
        path = _local_path(object_name)
        await asyncio.to_thread(path.unlink, True)  # missing_ok=True
        return

    client = _client_or_none()
    if client is None:
        return
    await client.remove_object(settings.minio_bucket_name, object_name)
