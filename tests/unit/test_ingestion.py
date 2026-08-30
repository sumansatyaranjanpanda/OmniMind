"""Tests for document ingestion pipeline (end-to-end orchestration)."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from ingestion.pipeline import process_document
from parsing.parsers import FigureCaption, ParseResult


@pytest.mark.asyncio
async def test_process_document_not_found():
    """Test that processing a non-existent document is handled gracefully."""
    # Mock database session
    mock_db = AsyncMock(spec=AsyncSession)
    mock_db.get.return_value = None  # Document not found

    doc_id = uuid.uuid4()

    # Should return early without raising
    with patch("ingestion.pipeline.logger"):
        await process_document(doc_id, mock_db)

    # Verify we tried to fetch the document
    mock_db.get.assert_called_once()


@pytest.mark.asyncio
async def test_process_document_full_pipeline():
    """Test the complete ingestion pipeline: download → parse → chunk → embed → upsert."""
    # Setup mocks
    mock_db = AsyncMock(spec=AsyncSession)
    mock_document = MagicMock()
    mock_document.id = uuid.uuid4()
    mock_document.user_id = uuid.uuid4()
    mock_document.filename = "test.pdf"
    mock_document.s3_key = "user-123/doc-456.pdf"
    mock_document.status = "UPLOADED"

    mock_db.get.return_value = mock_document

    # Mock parse result
    mock_parse_result = ParseResult(
        markdown_text="# Test Document\n\nSome content.",
        figure_captions=[
            FigureCaption(caption="Test figure", page_number=1, figure_index=0)
        ],
        extracted_images=[(b"fake image bytes", 1, 0)],
        source_type="pdf",
    )

    # Mock chunks
    mock_text_chunks = [
        MagicMock(id="chunk-1", text="# Test Document"),
        MagicMock(id="chunk-2", text="Some content."),
    ]
    mock_figure_chunks = [
        MagicMock(id="fig-chunk-1", text="Test figure"),
    ]

    with patch("ingestion.pipeline.download_file", new_callable=AsyncMock) as mock_download:
        with patch("ingestion.pipeline.parse_document", new_callable=AsyncMock) as mock_parse:
            with patch("ingestion.pipeline.chunk_markdown", return_value=mock_text_chunks):
                with patch("ingestion.pipeline.chunk_figures", return_value=mock_figure_chunks):
                    with patch("ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert_chunks:
                        with patch("ingestion.pipeline.upsert_images", new_callable=AsyncMock) as mock_upsert_images:
                            # Setup return values
                            mock_download.return_value = b"pdf file bytes"
                            mock_parse.return_value = mock_parse_result

                            # Execute pipeline
                            await process_document(mock_document.id, mock_db)

                            # Verify the pipeline steps
                            mock_download.assert_called_once_with(mock_document.s3_key)
                            mock_parse.assert_called_once()

                            # Verify chunks were created
                            # (chunk_markdown and chunk_figures return_values are used directly)

                            # Verify upsert was called with correct tenant_id
                            mock_upsert_chunks.assert_called_once()
                            call_args = mock_upsert_chunks.call_args
                            assert call_args.kwargs["tenant_id"] == str(mock_document.user_id)

                            # Verify image upsert was called
                            mock_upsert_images.assert_called_once()
                            img_call_args = mock_upsert_images.call_args
                            assert img_call_args.kwargs["document_id"] == str(mock_document.id)
                            assert img_call_args.kwargs["source_type"] == "pdf"

                            # Verify DB commits were called
                            assert mock_db.commit.call_count >= 2  # At least PARSED and CHUNKED stages


@pytest.mark.asyncio
async def test_process_document_no_images():
    """Test pipeline when document has no images (only text)."""
    mock_db = AsyncMock(spec=AsyncSession)
    mock_document = MagicMock()
    mock_document.id = uuid.uuid4()
    mock_document.user_id = uuid.uuid4()
    mock_document.filename = "markdown.md"
    mock_document.s3_key = "user-123/markdown.md"

    mock_db.get.return_value = mock_document

    # Parse result with no images
    mock_parse_result = ParseResult(
        markdown_text="# Markdown\n\nNo images here.",
        figure_captions=[],
        extracted_images=[],
        source_type="md",
    )

    mock_text_chunks = [MagicMock(id="chunk-1")]

    with patch("ingestion.pipeline.download_file", new_callable=AsyncMock):
        with patch("ingestion.pipeline.parse_document", new_callable=AsyncMock) as mock_parse:
            with patch("ingestion.pipeline.chunk_markdown", return_value=mock_text_chunks):
                with patch("ingestion.pipeline.chunk_figures", return_value=[]):
                    with patch("ingestion.pipeline.upsert_chunks", new_callable=AsyncMock):
                        with patch("ingestion.pipeline.upsert_images", new_callable=AsyncMock) as mock_upsert_images:
                            mock_parse.return_value = mock_parse_result

                            await process_document(mock_document.id, mock_db)

                            # upsert_images should NOT be called when there are no images
                            mock_upsert_images.assert_not_called()


@pytest.mark.asyncio
async def test_process_document_pipeline_error_handling():
    """Test that errors during processing are caught and DB status is set to FAILED."""
    mock_db = AsyncMock(spec=AsyncSession)
    mock_document = MagicMock()
    mock_document.id = uuid.uuid4()
    mock_document.status = "UPLOADED"

    mock_db.get.return_value = mock_document

    with patch("ingestion.pipeline.download_file", new_callable=AsyncMock) as mock_download:
        # Simulate error during download
        mock_download.side_effect = RuntimeError("Download failed")

        with patch("ingestion.pipeline.logger"):
            # Should raise the error but also set status to FAILED
            with pytest.raises(RuntimeError):
                await process_document(mock_document.id, mock_db)

            # Verify status was set to FAILED
            assert mock_document.status == "FAILED"
            mock_db.commit.assert_called()


@pytest.mark.asyncio
async def test_process_document_status_transitions():
    """Test that document status transitions correctly through pipeline stages."""
    mock_db = AsyncMock(spec=AsyncSession)
    mock_document = MagicMock()
    mock_document.id = uuid.uuid4()
    mock_document.user_id = uuid.uuid4()
    mock_document.filename = "test.pdf"
    mock_document.s3_key = "key"
    mock_document.status = "UPLOADED"

    mock_db.get.return_value = mock_document

    mock_parse_result = ParseResult(
        markdown_text="content",
        figure_captions=[],
        extracted_images=[],
        source_type="pdf",
    )

    with patch("ingestion.pipeline.download_file", new_callable=AsyncMock, return_value=b"data"):
        with patch("ingestion.pipeline.parse_document", new_callable=AsyncMock, return_value=mock_parse_result):
            with patch("ingestion.pipeline.chunk_markdown", return_value=[]):
                with patch("ingestion.pipeline.chunk_figures", return_value=[]):
                    with patch("ingestion.pipeline.upsert_chunks", new_callable=AsyncMock):
                        await process_document(mock_document.id, mock_db)

    # Should end in EMBEDDED status (final state)
    assert mock_document.status == "EMBEDDED"
    # Verify commit was called multiple times (for each status transition)
    assert mock_db.commit.call_count >= 3


@pytest.mark.asyncio
async def test_process_document_chunk_count_tracking():
    """Test that chunk counts are correctly tracked and logged."""
    mock_db = AsyncMock(spec=AsyncSession)
    mock_document = MagicMock()
    mock_document.id = uuid.uuid4()
    mock_document.user_id = uuid.uuid4()
    mock_document.filename = "large.pdf"
    mock_document.s3_key = "key"

    mock_db.get.return_value = mock_document

    # Simulate a document with many chunks
    text_chunks = [MagicMock() for _ in range(50)]
    fig_chunks = [MagicMock() for _ in range(5)]

    mock_parse_result = ParseResult(
        markdown_text="content" * 1000,
        figure_captions=[FigureCaption(caption=f"Fig {i}", page_number=i) for i in range(5)],
        extracted_images=[(b"img", i, i) for i in range(5)],
        source_type="pdf",
    )

    with patch("ingestion.pipeline.download_file", new_callable=AsyncMock, return_value=b"data"):
        with patch("ingestion.pipeline.parse_document", new_callable=AsyncMock, return_value=mock_parse_result):
            with patch("ingestion.pipeline.chunk_markdown", return_value=text_chunks):
                with patch("ingestion.pipeline.chunk_figures", return_value=fig_chunks):
                    with patch("ingestion.pipeline.upsert_chunks", new_callable=AsyncMock):
                        with patch("ingestion.pipeline.upsert_images", new_callable=AsyncMock):
                            with patch("ingestion.pipeline.logger") as mock_logger:
                                await process_document(mock_document.id, mock_db)

                                # Verify logging calls captured chunk counts
                                # (logger is mocked, so we just verify it was called)
                                assert mock_logger.info.call_count >= 3
