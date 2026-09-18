"""Router for document management and ingestion."""

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import async_session_factory, get_db
from api.deps import get_current_user
from api.models.document import Document
from api.models.user import User
from api.schemas.ingestion import DocumentResponse
from agents import corpus_manifest
from api.storage import delete_file, upload_file
from ingestion.pipeline import process_document
from retrieval.pinecone_client import delete_document_vectors

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


async def background_process_document(document_id: uuid.UUID) -> None:
    """Wrapper to run the ingestion pipeline with a fresh DB session."""
    async with async_session_factory() as session:
        await process_document(document_id, session)
        # The query analyzer routes on the list of ingested filenames, so a document
        # that just finished embedding has to become visible to it promptly — otherwise
        # the first questions about a fresh upload get routed as if it were never added.
        doc = await session.get(Document, document_id)
        if doc:
            corpus_manifest.invalidate(str(doc.user_id))


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document_endpoint(
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
):
    """Upload a document and queue it for processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing.")
        
    # Generate unique MinIO key
    doc_uuid = uuid.uuid4()
    ext = file.filename.split(".")[-1] if "." in file.filename else ""
    s3_key = f"{current_user.id}/{doc_uuid}.{ext}" if ext else f"{current_user.id}/{doc_uuid}"
    
    # Read file bytes
    file_bytes = await file.read()
    
    # Upload to MinIO
    await upload_file(s3_key, file_bytes, content_type=file.content_type or "application/octet-stream")
    
    # Create DB record
    document = Document(
        id=doc_uuid,
        user_id=current_user.id,
        filename=file.filename,
        s3_key=s3_key,
        status="UPLOADED",
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)
    
    # Queue background task
    background_tasks.add_task(background_process_document, document.id)
    
    return document


@router.post("/{document_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document_endpoint(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Reprocess an existing document."""
    doc = await db.get(Document, document_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc.status = "UPLOADED"
    await db.commit()
    await db.refresh(doc)

    background_tasks.add_task(background_process_document, doc.id)
    return doc


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document_endpoint(
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Fetch a single document's current status — for the client to poll while a
    just-uploaded document moves through UPLOADED -> PARSED -> CHUNKED -> EMBEDDED
    (or FAILED), so the UI can show real per-stage progress instead of a fixed-delay
    guess at when ingestion is done.
    """
    doc = await db.get(Document, document_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """List all documents for the current user."""
    stmt = select(Document).where(Document.user_id == current_user.id).order_by(Document.created_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document_endpoint(
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a document and everything derived from it: the file, its vector
    embeddings, and its database record. Storage and vector cleanup are
    best-effort — a hiccup there shouldn't leave the user unable to remove a
    document from their catalog, so failures are logged and the delete proceeds.
    """
    doc = await db.get(Document, document_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        await delete_file(doc.s3_key)
    except Exception as e:
        logger.warning("Failed to delete document file from storage", document_id=str(document_id), error=str(e))

    try:
        await delete_document_vectors(str(document_id), tenant_id=str(current_user.id))
    except Exception as e:
        logger.warning("Failed to delete document vectors", document_id=str(document_id), error=str(e))

    await db.delete(doc)
    await db.commit()
    corpus_manifest.invalidate(str(current_user.id))
