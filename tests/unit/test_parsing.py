"""Tests for multimodal document parsing (Docling + Gemini vision)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parsing.parsers import FigureCaption, ParseResult, parse_document


def test_figure_caption_dataclass():
    """Verify FigureCaption instantiation and defaults."""
    caption = FigureCaption(
        caption="A line chart showing trends",
        page_number=5,
        figure_index=0,
    )
    assert caption.caption == "A line chart showing trends"
    assert caption.page_number == 5
    assert caption.figure_index == 0
    assert caption.content_type == "figure_caption"


def test_parse_result_dataclass():
    """Verify ParseResult instantiation and defaults."""
    result = ParseResult(
        markdown_text="# Heading\n\nParagraph.",
        source_type="pdf",
    )
    assert result.markdown_text == "# Heading\n\nParagraph."
    assert result.source_type == "pdf"
    assert result.figure_captions == []
    assert result.extracted_images == []


@pytest.mark.asyncio
async def test_parse_document_plaintext():
    """Test fast path: plaintext file (no Docling, no vision)."""
    file_bytes = b"This is plain text content.\nNo formatting."
    filename = "test.txt"

    result = await parse_document(file_bytes, filename)

    assert result.markdown_text == "This is plain text content.\nNo formatting."
    assert result.figure_captions == []
    assert result.extracted_images == []
    assert result.source_type == "txt"


@pytest.mark.asyncio
async def test_parse_document_markdown():
    """Test fast path: Markdown file."""
    markdown_content = "# Title\n\nSome **bold** text."
    file_bytes = markdown_content.encode("utf-8")
    filename = "readme.md"

    result = await parse_document(file_bytes, filename)

    assert result.markdown_text == markdown_content
    assert result.figure_captions == []
    assert result.source_type == "md"


@pytest.mark.asyncio
async def test_parse_document_csv():
    """Test fast path: CSV file."""
    csv_content = "Name,Age,City\nAlice,30,NYC\nBob,25,LA"
    file_bytes = csv_content.encode("utf-8")
    filename = "data.csv"

    result = await parse_document(file_bytes, filename)

    assert result.markdown_text == csv_content
    assert result.source_type == "csv"


@pytest.mark.asyncio
async def test_unparseable_pdf_raises_rather_than_decoding_bytes_as_text():
    """A .pdf that the real parser cannot read must fail, not fall back.

    This test previously asserted the opposite — it fed markdown bytes under a .pdf
    name and expected the decode fallback to return them, with a docstring conceding
    it was a stand-in because "Docling mocking is complex". That silent fallback is
    what shipped the 2026-09-18 bug: a torch/torchvision ABI mismatch broke Docling
    in the container, every PDF was decoded as raw bytes, and ingestion reported
    success while storing 173 chunks of mojibake. Nothing surfaced the failure until
    queries started answering "insufficient evidence" for documents that were right
    there. Failing loudly is the behaviour worth locking in.
    """
    with pytest.raises(Exception) as exc:
        await parse_document(b"# Not actually a PDF\n\nSome text.", "document.pdf")

    assert "pdf" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_parse_document_docling_error_fallback():
    """Test that parsing falls back to text decoding if Docling fails.

    Note: This test verifies the fallback is used, but because Docling
    mocking is complex, we test it more indirectly via the text fast path.
    """
    # Test the direct text fallback path
    file_bytes = "Fallback text content".encode("utf-8")
    filename = "document.txt"

    result = await parse_document(file_bytes, filename)

    # Should use text directly
    assert result.markdown_text == "Fallback text content"
    assert result.figure_captions == []


@pytest.mark.asyncio
async def test_parse_document_multiple_figures():
    """Test that ParseResult can hold multiple figure captions.

    Note: Full Docling mocking with concurrent Gemini calls is complex.
    This test verifies the data structure supports it; integration tests
    should verify with real PDFs.
    """
    captions = [
        FigureCaption(caption="Chart showing sales data.", page_number=1, figure_index=0),
        FigureCaption(caption="Diagram of system architecture.", page_number=2, figure_index=1),
        FigureCaption(caption="Photo of the team.", page_number=3, figure_index=2),
    ]

    result = ParseResult(
        markdown_text="# Document\n\nWith images.",
        figure_captions=captions,
        source_type="pdf",
    )

    assert len(result.figure_captions) == 3
    assert result.figure_captions[0].caption == "Chart showing sales data."
    assert result.figure_captions[1].page_number == 2


@pytest.mark.asyncio
async def test_parse_document_invalid_utf8():
    """Test that invalid UTF-8 in plaintext is handled gracefully."""
    # Invalid UTF-8 bytes
    file_bytes = b"Valid text\xff\xfe Invalid bytes"
    filename = "corrupted.txt"

    result = await parse_document(file_bytes, filename)

    # Should decode with errors='ignore'
    assert "Valid text" in result.markdown_text
    assert result.source_type == "txt"


# ── Binary-format parse failures must be loud (added 2026-09-18) ──
#
# A torch/torchvision ABI mismatch broke Docling's layout model inside the container.
# The old fallback decoded the PDF's raw bytes as UTF-8, so ingestion "succeeded"
# with 173 chunks of mojibake that embedded and retrieved fine but meant nothing.
# The only visible symptom was an unexplained "insufficient evidence" at query time.


def test_pdf_parse_failure_raises_instead_of_storing_garbage():
    from parsing.parsers import _docling_convert_sync

    # No docling importable in this test env -> the failure path under test.
    with pytest.raises(Exception) as exc:
        _docling_convert_sync(b"%PDF-1.7 \x00\x81\xff binary bytes", "report.pdf")

    assert "pdf" in str(exc.value).lower()


def test_office_formats_are_treated_the_same_way():
    from parsing.parsers import _BINARY_DOCUMENT_FORMATS

    for ext in ("pdf", "docx", "pptx", "xlsx"):
        assert ext in _BINARY_DOCUMENT_FORMATS


def test_markdown_still_falls_back_to_text_decoding():
    """For text formats the decode fallback is a real answer, so it stays."""
    from parsing.parsers import _docling_convert_sync

    text, figures = _docling_convert_sync(b"# Title\n\nPlain body.", "notes.md")
    assert "Plain body." in text
    assert figures == []
