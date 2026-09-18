"""Langfuse SDK client initialization and no-op safety fallback."""

from __future__ import annotations

from typing import Any

import structlog

from api.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

_langfuse_client: Any = None
_initialized: bool = False


class NullSpan:
    """No-op span object for when Langfuse is not enabled."""

    def __init__(self, name: str = "null_span"):
        self.name = name

    def end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        pass

    def span(self, name: str, *args: Any, **kwargs: Any) -> NullSpan:
        return NullSpan(name)

    def generation(self, name: str, *args: Any, **kwargs: Any) -> NullSpan:
        return NullSpan(name)

    def event(self, name: str, *args: Any, **kwargs: Any) -> None:
        pass

    def score(self, *args: Any, **kwargs: Any) -> None:
        pass


class NullTrace(NullSpan):
    """No-op trace object mimicking Langfuse trace."""

    def __init__(self, id: str = "null_trace", name: str = "null_trace"):
        super().__init__(name)
        self.id = id


def get_langfuse_client() -> Any:
    """Returns the cached Langfuse client, or None if credentials are not configured."""
    global _langfuse_client, _initialized
    if not _initialized:
        current_settings = get_settings()
        pk = (current_settings.langfuse_public_key or "").strip('"\' ')
        sk = (current_settings.langfuse_secret_key or "").strip('"\' ')
        host = (current_settings.langfuse_base_url or current_settings.langfuse_host or "https://cloud.langfuse.com").strip('"\' ')

        if pk and sk:
            try:
                from langfuse import Langfuse

                logger.info(
                    "Initializing Langfuse client",
                    host=host,
                    public_key=pk[:8] + "...",
                )
                _langfuse_client = Langfuse(
                    public_key=pk,
                    secret_key=sk,
                    host=host,
                )
            except Exception as e:
                logger.warning("Failed to initialize Langfuse SDK", error=str(e))
                _langfuse_client = None
        else:
            logger.debug("Langfuse credentials not configured; tracing will use local structured logs")
            _langfuse_client = None
        _initialized = True
    return _langfuse_client


def flush_traces() -> None:
    """Flush pending traces to the Langfuse server."""
    client = get_langfuse_client()
    if client and hasattr(client, "flush"):
        try:
            client.flush()
        except Exception as e:
            logger.warning("Failed to flush Langfuse traces", error=str(e))
