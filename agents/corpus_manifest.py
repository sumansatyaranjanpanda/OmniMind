"""Corpus manifest — what documents does this tenant actually have?

The query analyzer used to route purely on the wording of the question, with no idea
what was in the index. That is why "who is abinash" was classified as a public-figure
lookup and sent to web search while `AbinashResume.pdf` sat in the corpus, unread: the
router had no way to know the answer was already there.

Feeding it the list of ingested filenames turns an unanswerable guess ("is this person
internal or public?") into a lookup ("is there a document that plausibly covers this?").

Cached per tenant with a short TTL because it sits on the hot path and changes only on
upload/delete — a stale entry costs at most one query routed on a slightly old view of
the corpus, while querying Postgres on every turn costs latency on every turn.
"""

from __future__ import annotations

import time

import structlog
from sqlalchemy import select

logger = structlog.get_logger(__name__)

# tenant_id -> (expires_at_monotonic, filenames)
_CACHE: dict[str, tuple[float, list[str]]] = {}
_TTL_SECONDS = 60.0

# Enough for the model to recognise topics without crowding out the prompt.
_MAX_DOCUMENTS = 40


def invalidate(tenant_id: str) -> None:
    """Drop the cached manifest — call after an upload or delete."""
    _CACHE.pop(tenant_id, None)


async def get_corpus_documents(tenant_id: str) -> list[str]:
    """Filenames of this tenant's successfully embedded documents.

    Returns an empty list on any failure. Routing degrades to the old
    corpus-blind behaviour rather than failing the request — a slightly worse
    route is always better than no answer.
    """
    now = time.monotonic()
    cached = _CACHE.get(tenant_id)
    if cached and cached[0] > now:
        return cached[1]

    try:
        from uuid import UUID

        from api.database import async_session_factory
        from api.models.document import Document

        async with async_session_factory() as session:
            result = await session.execute(
                select(Document.filename)
                .where(Document.user_id == UUID(tenant_id))
                .where(Document.status == "EMBEDDED")
                .order_by(Document.created_at.desc())
                .limit(_MAX_DOCUMENTS)
            )
            filenames = [row[0] for row in result if row[0]]
    except Exception as exc:
        # Includes the case where tenant_id isn't a UUID (tests, demo tenants).
        logger.debug("Corpus manifest unavailable", tenant_id=tenant_id, error=str(exc))
        return []

    _CACHE[tenant_id] = (now + _TTL_SECONDS, filenames)
    return filenames
