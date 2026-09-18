"""Re-run ingestion for documents already stored in MinIO.

Chunking changes (dropping heading-only chunks, adding document/section context to each
chunk) only affect documents ingested *after* the change. Anything already in Pinecone
keeps the chunks it was created with, so without this the improvement is invisible on
exactly the documents being used to test it.

Old vectors are deleted first — chunk IDs are content-derived, so re-running ingestion
alone would leave the previous chunks orphaned in the namespace rather than replacing them.

    python -m scripts.reindex_documents            # every EMBEDDED document
    python -m scripts.reindex_documents --user <uuid>
    python -m scripts.reindex_documents --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import structlog
from sqlalchemy import select

from api.database import async_session_factory
from api.models.document import Document
from ingestion.pipeline import process_document
from retrieval.pinecone_client import delete_document_vectors

logger = structlog.get_logger(__name__)


async def reindex(user_id: str | None, dry_run: bool) -> int:
    async with async_session_factory() as session:
        stmt = select(Document).where(Document.status == "EMBEDDED")
        if user_id:
            from uuid import UUID

            stmt = stmt.where(Document.user_id == UUID(user_id))
        documents = list((await session.execute(stmt)).scalars())

    if not documents:
        print("No EMBEDDED documents found.")
        return 0

    print(f"{'Would re-index' if dry_run else 'Re-indexing'} {len(documents)} document(s):")
    for doc in documents:
        print(f"  - {doc.filename} (chunks: {doc.chunk_count}, user: {doc.user_id})")
    if dry_run:
        return 0

    failures = 0
    for doc in documents:
        tenant_id = str(doc.user_id)
        try:
            await delete_document_vectors(str(doc.id), tenant_id=tenant_id)
        except Exception as exc:
            # Not fatal: re-ingestion will still write the new chunks. Stale chunks may
            # linger in the namespace, which costs ranking quality but not correctness.
            logger.warning("Could not delete old vectors", document=doc.filename, error=str(exc))

        try:
            async with async_session_factory() as session:
                await process_document(doc.id, session)
            async with async_session_factory() as session:
                refreshed = await session.get(Document, doc.id)
                count = refreshed.chunk_count if refreshed else "?"
            print(f"  OK   {doc.filename} -> {count} chunks")
        except Exception as exc:
            failures += 1
            print(f"  FAIL {doc.filename}: {exc}")

    from agents import corpus_manifest

    for doc in documents:
        corpus_manifest.invalidate(str(doc.user_id))

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", help="Only re-index documents owned by this user UUID")
    parser.add_argument("--dry-run", action="store_true", help="List what would be re-indexed")
    args = parser.parse_args()

    failures = asyncio.run(reindex(args.user, args.dry_run))
    if failures:
        print(f"\n{failures} document(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
