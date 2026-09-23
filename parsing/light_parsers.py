"""Dependency-light document text extraction.

Used when Docling is not installed. Docling is the better parser — it is layout
aware, keeps tables as tables and can hand back figures for captioning — but it
pulls PyTorch, transformers and timm behind it, which is roughly 3GB of image and
several hundred MB of resident memory. That is affordable on a VPS and not
affordable on a 512MB free tier, so the deploy image ships without it and lands
here instead.

The tradeoff is explicit: these extractors get the *text* out faithfully, but a
table arrives as its cell text rather than a structured table, and figures are not
extracted at all (so nothing is sent for vision captioning). Retrieval, citations,
the critic and every other stage behave exactly the same — they just see plainer
source text.

Install the heavy path with `pip install -e ".[full]"` to get Docling back.
"""

from __future__ import annotations

import io

import structlog

logger = structlog.get_logger(__name__)


def _pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            # Page markers double as provenance anchors: chunking keeps them in the
            # chunk text, so a citation can still say which page a claim came from.
            pages.append(f"\n\n## Page {i}\n\n{text}")
    return "".join(pages).strip()


def _docx(file_bytes: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(file_bytes))
    parts: list[str] = []

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # Word's built-in heading styles are the only structure signal available
        # here, and the markdown chunker splits on headings — so preserving them
        # is what keeps sections from collapsing into one undifferentiated blob.
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading"):
            level = "".join(c for c in style if c.isdigit()) or "2"
            parts.append(f"\n\n{'#' * min(int(level), 6)} {text}\n")
        else:
            parts.append(text)

    for table in document.tables:
        rows = [
            " | ".join(cell.text.strip() for cell in row.cells)
            for row in table.rows
            if any(cell.text.strip() for cell in row.cells)
        ]
        if rows:
            parts.append("\n\n" + "\n".join(rows))

    return "\n\n".join(parts).strip()


def _pptx(file_bytes: bytes) -> str:
    from pptx import Presentation

    prs = Presentation(io.BytesIO(file_bytes))
    slides: list[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        lines = [
            shape.text.strip()
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and shape.text.strip()
        ]
        if lines:
            slides.append(f"\n\n## Slide {i}\n\n" + "\n\n".join(lines))
    return "".join(slides).strip()


def _xlsx(file_bytes: bytes) -> str:
    from openpyxl import load_workbook

    # read_only + data_only: stream rather than build the whole object graph, and
    # take computed values rather than formula strings — "=SUM(A1:A9)" is not
    # something anyone can ask a question about.
    wb = load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    sheets: list[str] = []
    for ws in wb.worksheets:
        rows: list[str] = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            sheets.append(f"\n\n## Sheet: {ws.title}\n\n" + "\n".join(rows))
    wb.close()
    return "".join(sheets).strip()


def _html(file_bytes: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(file_bytes, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n").strip()


_EXTRACTORS = {
    "pdf": _pdf,
    "docx": _docx,
    "doc": _docx,
    "pptx": _pptx,
    "ppt": _pptx,
    "xlsx": _xlsx,
    "xls": _xlsx,
    "html": _html,
    "htm": _html,
}


def can_parse(ext: str) -> bool:
    return ext.lower() in _EXTRACTORS


def extract_text(file_bytes: bytes, ext: str) -> str:
    """Extract text for a known format. Raises if the format is unsupported.

    Raising rather than returning "" on failure is deliberate and matches the
    Docling path: a binary document that silently yields empty text would be
    ingested as an empty document and surface much later as an unexplained
    "insufficient evidence" at query time.
    """
    ext = ext.lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        raise ValueError(f"No lightweight extractor for .{ext}")

    text = extractor(file_bytes)
    if not text.strip():
        raise RuntimeError(
            f"Extracted no text from .{ext} document — it may be a scanned image "
            f"(OCR is only available on the Docling path)"
        )
    return text
