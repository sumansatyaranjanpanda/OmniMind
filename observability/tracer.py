"""Tracing context managers and decorators for OmniMind RAG pipeline."""

from __future__ import annotations

import asyncio
import functools
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, TypeVar

import structlog

from observability.langfuse_client import NullSpan, NullTrace, flush_traces, get_langfuse_client
from observability.metrics import calculate_cost, estimate_tokens

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class TraceContext:
    """Manages the lifecycle of a distributed trace across RAG stages."""

    def __init__(
        self,
        name: str,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.name = name
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.start_time = time.perf_counter()

        client = get_langfuse_client()
        self.root_span = None

        if client:
            try:
                # In Langfuse v4, start root observation
                self.root_span = client.start_observation(
                    name=self.name,
                    as_type="span",
                    input={"trace_id": self.trace_id, "user_id": self.user_id, **self.metadata},
                    metadata={"user_id": self.user_id, "session_id": self.session_id, **self.metadata},
                )
            except Exception as e:
                logger.warning("Failed to start Langfuse root observation", error=str(e))
                self.root_span = NullSpan(name)
        else:
            self.root_span = NullSpan(name)

    @asynccontextmanager
    async def span(
        self,
        name: str,
        input_data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Create a child span within the current trace."""
        start = time.perf_counter()
        meta = metadata or {}
        client = get_langfuse_client()

        langfuse_span = None
        if client:
            try:
                langfuse_span = client.start_observation(
                    name=name,
                    as_type="span",
                    input=input_data,
                    metadata={"trace_id": self.trace_id, "user_id": self.user_id, **meta},
                )
            except Exception:
                langfuse_span = NullSpan(name)
        else:
            langfuse_span = NullSpan(name)

        error: Exception | None = None
        output_data: Any = None

        try:
            yield langfuse_span
        except Exception as exc:
            error = exc
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            status = "ERROR" if error else "SUCCESS"

            if langfuse_span and hasattr(langfuse_span, "end"):
                try:
                    if hasattr(langfuse_span, "update"):
                        langfuse_span.update(
                            output=output_data,
                            status_message=str(error) if error else None,
                            level="ERROR" if error else "DEFAULT",
                        )
                    langfuse_span.end()
                except Exception:
                    pass

            logger.info(
                "Trace span completed",
                trace_id=self.trace_id,
                span_name=name,
                latency_ms=elapsed_ms,
                status=status,
                user_id=self.user_id,
                **meta,
            )

    @asynccontextmanager
    async def generation(
        self,
        name: str,
        model: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Instrument an LLM generation call with token & cost tracking."""
        start = time.perf_counter()
        meta = metadata or {}
        input_tokens = estimate_tokens(prompt)

        gen_data: dict[str, Any] = {
            "output_text": "",
            "model": model,
        }

        client = get_langfuse_client()
        langfuse_gen = None

        if client:
            try:
                langfuse_gen = client.start_observation(
                    name=name,
                    as_type="generation",
                    input=prompt,
                    model=model,
                    metadata={"trace_id": self.trace_id, "user_id": self.user_id, **meta},
                )
            except Exception:
                langfuse_gen = NullSpan(name)
        else:
            langfuse_gen = NullSpan(name)

        error: Exception | None = None
        try:
            yield gen_data
        except Exception as exc:
            error = exc
            raise
        finally:
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
            output_tokens = estimate_tokens(gen_data.get("output_text", ""))
            cost = calculate_cost(model, input_tokens=input_tokens, output_tokens=output_tokens)

            if langfuse_gen and hasattr(langfuse_gen, "end"):
                try:
                    if hasattr(langfuse_gen, "update"):
                        langfuse_gen.update(
                            output=gen_data.get("output_text", ""),
                            usage={
                                "input": input_tokens,
                                "output": output_tokens,
                                "total": input_tokens + output_tokens,
                                "unit": "TOKENS",
                            },
                            status_message=str(error) if error else None,
                            level="ERROR" if error else "DEFAULT",
                        )
                    langfuse_gen.end()
                except Exception:
                    pass

            logger.info(
                "Generation completed",
                trace_id=self.trace_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=elapsed_ms,
                status="ERROR" if error else "SUCCESS",
            )


def observe(span_name: str | None = None) -> Callable[[F], F]:
    """Decorator to trace any async function execution."""

    def decorator(func: F) -> F:
        name = span_name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            error: Exception | None = None
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                error = exc
                raise
            finally:
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                logger.debug(
                    "Observed function call",
                    function=name,
                    latency_ms=elapsed_ms,
                    status="ERROR" if error else "SUCCESS",
                )

        return wrapper  # type: ignore[return-value]

    return decorator
