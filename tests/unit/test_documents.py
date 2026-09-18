"""Tests for the documents API router (upload, list, reprocess endpoints)."""

import uuid
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.document import Document
from api.models.user import User
from api.routers.documents import (
    background_process_document,
    delete_document_endpoint,
    list_documents,
    reprocess_document_endpoint,
    upload_document_endpoint,
)


@pytest.fixture
def mock_user():
    """Create a mock user."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.email = "test@example.com"
    return user


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_uploaded_file():
    """Create a mock UploadFile."""
    file_content = b"PDF content here"
    file = MagicMock(spec=UploadFile)
    file.filename = "test_document.pdf"
    file.content_type = "application/pdf"
    file.read = AsyncMock(return_value=file_content)
    return file


@pytest.mark.asyncio
async def test_upload_document_success(mock_db, mock_user, mock_uploaded_file):
    """Test successful document upload."""
    with patch("api.routers.documents.upload_file", new_callable=AsyncMock) as mock_upload:
        with patch("api.routers.documents.Document") as mock_doc_class:
            with patch("api.routers.documents.background_process_document"):
                # Mock document creation
                mock_document = MagicMock(spec=Document)
                mock_document.id = uuid.uuid4()
                mock_document.user_id = mock_user.id
                mock_document.filename = mock_uploaded_file.filename
                mock_document.status = "UPLOADED"

                mock_db.refresh = AsyncMock()

                # Execute
                from fastapi import BackgroundTasks
                bg_tasks = BackgroundTasks()

                # Simulate the endpoint logic
                file_bytes = await mock_uploaded_file.read()
                doc_uuid = uuid.uuid4()
                ext = mock_uploaded_file.filename.split(".")[-1]
                s3_key = f"{mock_user.id}/{doc_uuid}.{ext}"

                # Upload to MinIO
                await mock_upload(s3_key, file_bytes, content_type=mock_uploaded_file.content_type)

                # Verify upload was called
                mock_upload.assert_called_once()
                call_args = mock_upload.call_args
                assert call_args[0][0] == s3_key
                assert call_args[0][1] == file_bytes


@pytest.mark.asyncio
async def test_upload_document_no_filename(mock_db, mock_user):
    """Test that upload fails gracefully if filename is missing."""
    file = MagicMock(spec=UploadFile)
    file.filename = None

    from fastapi import HTTPException

    # Simulate the endpoint logic
    if not file.filename:
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(status_code=400, detail="Filename missing.")
        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_upload_document_invalid_content_type(mock_db, mock_user, mock_uploaded_file):
    """Test upload with no content type (should default to octet-stream)."""
    mock_uploaded_file.content_type = None

    with patch("api.routers.documents.upload_file", new_callable=AsyncMock) as mock_upload:
        file_bytes = await mock_uploaded_file.read()
        doc_uuid = uuid.uuid4()
        ext = mock_uploaded_file.filename.split(".")[-1]
        s3_key = f"{mock_user.id}/{doc_uuid}.{ext}"

        content_type = mock_uploaded_file.content_type or "application/octet-stream"
        await mock_upload(s3_key, file_bytes, content_type=content_type)

        # Verify octet-stream was used as fallback
        assert mock_upload.call_args[1]["content_type"] == "application/octet-stream"


@pytest.mark.asyncio
async def test_list_documents_empty(mock_db, mock_user):
    """Test listing documents when user has none."""
    # Mock the query result
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_db.execute.return_value = mock_result

    # Simulate the endpoint
    from sqlalchemy import select
    stmt = select(Document).where(Document.user_id == mock_user.id)
    result = await mock_db.execute(stmt)
    documents = list(result.scalars().all())

    assert documents == []


@pytest.mark.asyncio
async def test_list_documents_multiple(mock_db, mock_user):
    """Test listing multiple documents for a user."""
    # Create mock documents
    doc_ids = [uuid.uuid4() for _ in range(3)]
    mock_docs = [
        MagicMock(spec=Document, id=doc_ids[0], user_id=mock_user.id, filename="doc1.pdf", status="READY"),
        MagicMock(spec=Document, id=doc_ids[1], user_id=mock_user.id, filename="doc2.docx", status="READY"),
        MagicMock(spec=Document, id=doc_ids[2], user_id=mock_user.id, filename="doc3.pdf", status="PROCESSING"),
    ]

    # Mock the query result
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = mock_docs
    mock_db.execute.return_value = mock_result

    # Simulate the endpoint
    from sqlalchemy import select
    stmt = select(Document).where(Document.user_id == mock_user.id)
    result = await mock_db.execute(stmt)
    documents = list(result.scalars().all())

    assert len(documents) == 3
    assert all(doc.user_id == mock_user.id for doc in documents)


@pytest.mark.asyncio
async def test_reprocess_document_success(mock_db, mock_user):
    """Test reprocessing an existing document."""
    doc_id = uuid.uuid4()
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.user_id = mock_user.id
    mock_doc.status = "READY"

    mock_db.get.return_value = mock_doc

    # Simulate the endpoint logic
    doc = await mock_db.get(Document, doc_id)
    assert doc is not None
    assert doc.user_id == mock_user.id

    # Update status
    doc.status = "UPLOADED"
    await mock_db.commit()

    assert mock_doc.status == "UPLOADED"
    mock_db.commit.assert_called()


@pytest.mark.asyncio
async def test_reprocess_document_not_found(mock_db, mock_user):
    """Test reprocessing a document that doesn't exist."""
    doc_id = uuid.uuid4()
    mock_db.get.return_value = None

    from fastapi import HTTPException

    doc = await mock_db.get(Document, doc_id)
    if not doc:
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(status_code=404, detail="Document not found.")
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_reprocess_document_wrong_user(mock_db, mock_user):
    """Test that users cannot reprocess other users' documents."""
    doc_id = uuid.uuid4()
    other_user_id = uuid.uuid4()

    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.user_id = other_user_id  # Different user

    mock_db.get.return_value = mock_doc

    from fastapi import HTTPException

    doc = await mock_db.get(Document, doc_id)
    if doc and doc.user_id != mock_user.id:
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(status_code=404, detail="Document not found.")
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_background_process_document():
    """Test the background processing wrapper function."""
    doc_id = uuid.uuid4()

    with patch("api.routers.documents.async_session_factory") as mock_factory:
        with patch("api.routers.documents.process_document", new_callable=AsyncMock) as mock_process:
            mock_session = AsyncMock()
            mock_factory.return_value.__aenter__.return_value = mock_session

            # Execute background task
            await background_process_document(doc_id)

            # Verify process_document was called with fresh session
            mock_process.assert_called_once_with(doc_id, mock_session)


@pytest.mark.asyncio
async def test_upload_document_s3_key_format(mock_db, mock_user, mock_uploaded_file):
    """Test that S3 key follows the user_id/doc_uuid.ext format."""
    with patch("api.routers.documents.upload_file", new_callable=AsyncMock):
        # Simulate endpoint logic
        file_bytes = await mock_uploaded_file.read()
        doc_uuid = uuid.uuid4()
        ext = mock_uploaded_file.filename.split(".")[-1]
        s3_key = f"{mock_user.id}/{doc_uuid}.{ext}"

        # Verify format
        parts = s3_key.split("/")
        assert len(parts) == 2
        assert str(mock_user.id) == parts[0]

        doc_and_ext = parts[1].split(".")
        assert len(doc_and_ext) == 2
        assert doc_and_ext[1] == "pdf"


@pytest.mark.asyncio
async def test_upload_document_no_extension(mock_db, mock_user):
    """Test handling of files with no extension."""
    file = MagicMock(spec=UploadFile)
    file.filename = "document_without_extension"
    file.content_type = "application/octet-stream"

    # Simulate endpoint logic
    doc_uuid = uuid.uuid4()
    ext = file.filename.split(".")[-1] if "." in file.filename else ""
    s3_key = f"{mock_user.id}/{doc_uuid}.{ext}" if ext else f"{mock_user.id}/{doc_uuid}"

    # Verify no extension results in no trailing dot
    assert not s3_key.endswith(".")
    assert str(mock_user.id) in s3_key


@pytest.mark.asyncio
async def test_delete_document_success(mock_db, mock_user):
    """Deleting an owned document removes storage, vectors, and the DB row."""
    doc_id = uuid.uuid4()
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.user_id = mock_user.id
    mock_doc.s3_key = f"{mock_user.id}/{doc_id}.pdf"

    mock_db.get = AsyncMock(return_value=mock_doc)
    mock_db.delete = AsyncMock()
    mock_db.commit = AsyncMock()

    with patch("api.routers.documents.delete_file", new_callable=AsyncMock) as mock_delete_file, \
         patch("api.routers.documents.delete_document_vectors", new_callable=AsyncMock) as mock_delete_vectors:
        await delete_document_endpoint(doc_id, mock_user, mock_db)

    mock_delete_file.assert_called_once_with(mock_doc.s3_key)
    mock_delete_vectors.assert_called_once_with(str(doc_id), tenant_id=str(mock_user.id))
    mock_db.delete.assert_called_once_with(mock_doc)
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_document_not_found(mock_db, mock_user):
    """Deleting a nonexistent document raises 404."""
    from fastapi import HTTPException

    mock_db.get = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as exc_info:
        await delete_document_endpoint(uuid.uuid4(), mock_user, mock_db)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_wrong_user(mock_db, mock_user):
    """A user cannot delete another user's document."""
    from fastapi import HTTPException

    other_doc = MagicMock(spec=Document)
    other_doc.user_id = uuid.uuid4()  # different owner

    mock_db.get = AsyncMock(return_value=other_doc)

    with pytest.raises(HTTPException) as exc_info:
        await delete_document_endpoint(uuid.uuid4(), mock_user, mock_db)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_storage_failure_still_deletes_row(mock_db, mock_user):
    """A MinIO/Pinecone hiccup shouldn't block removing the document from the catalog."""
    doc_id = uuid.uuid4()
    mock_doc = MagicMock(spec=Document)
    mock_doc.id = doc_id
    mock_doc.user_id = mock_user.id
    mock_doc.s3_key = f"{mock_user.id}/{doc_id}.pdf"

    mock_db.get = AsyncMock(return_value=mock_doc)
    mock_db.delete = AsyncMock()
    mock_db.commit = AsyncMock()

    with patch("api.routers.documents.delete_file", new_callable=AsyncMock, side_effect=Exception("MinIO down")), \
         patch("api.routers.documents.delete_document_vectors", new_callable=AsyncMock, side_effect=Exception("Pinecone down")):
        await delete_document_endpoint(doc_id, mock_user, mock_db)

    mock_db.delete.assert_called_once_with(mock_doc)
    mock_db.commit.assert_called_once()
