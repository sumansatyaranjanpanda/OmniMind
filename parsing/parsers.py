"""Multimodal document parser using Docling + Gemini 3.5 Flash-Lite.

Converts various document formats (PDF, DOCX, PPTX, XLSX, HTML, etc.) into
structured Markdown while also extracting and captioning images/diagrams/charts
using Google's Gemini 3.5 Flash-Lite vision model.

Architecture:
    1. Docling's DocumentConverter handles layout-aware parsing for ALL formats.
    2. Text, tables, and formulas are converted to structured Markdown.
    3. Images/figures/charts are detected, extracted, and sent to Gemini 3.5
       Flash-Lite for descriptive captioning.
    4. The output is a ParseResult containing both the Markdown text AND
       a list of figure captions with provenance.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Disable PyTorch TorchDynamo / TorchInductor compilation on Windows to prevent missing cl.exe error
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import structlog

from api.config import get_settings

logger = structlog.get_logger(__name__)

# Formats whose bytes are not text under any encoding. If the real parser fails on
# one of these there is no meaningful fallback, so ingestion must fail rather than
# store the decoded bytes — see the parse error path below.
_BINARY_DOCUMENT_FORMATS = {"pdf", "docx", "doc", "pptx", "ppt", "xlsx", "xls"}
settings = get_settings()

# Lazy-init Gemini client for vision captioning
_gemini_client = None


def _get_gemini_client():
    """Returns a cached Google GenAI client instance."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for multimodal document parsing. "
                "Set it in your .env file."
            )
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


@dataclass
class FigureCaption:
    """Represents a captioned image/figure extracted from a document."""

    caption: str
    page_number: int | None = None
    figure_index: int = 0
    content_type: str = "figure_caption"  # Could also be "chart", "diagram"


@dataclass
class ParseResult:
    """Output of the multimodal parser.

    Contains both the structured Markdown text AND any extracted figure captions.
    The pipeline can then chunk both independently and store them in the vector DB.
    """

    markdown_text: str
    figure_captions: list[FigureCaption] = field(default_factory=list)
    extracted_images: list[tuple[bytes, int | None, int]] = field(default_factory=list)
    source_type: str = ""


def _docling_convert_sync(file_bytes: bytes, filename: str) -> tuple[str, list[tuple[bytes, int | None, int]]]:
    """Synchronous Docling conversion (runs in a thread).

    Returns:
        Tuple of (markdown_text, list of (image_bytes, page_number, figure_index))
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    
    # Fast path for plain text and markdown
    if ext in ("md", "markdown", "txt", "csv", "json", "xml", "log", "yaml", "yml"):
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
            return text, []
        except Exception:
            pass

    # Docling needs a file path — write bytes to a temp file
    suffix = f".{ext}" if ext else ""

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(tmp_path)

            # Extract the Markdown representation
            markdown_text = result.document.export_to_markdown()

            # Extract figures/images
            figures: list[tuple[bytes, int | None, int]] = []
            fig_index = 0

            for item, _level in result.document.iterate_items():
                item_type = type(item).__name__

                if item_type == "PictureItem":
                    # Try to get the image bytes from the PictureItem
                    try:
                        image = item.get_image(result.document)
                        if image is not None:
                            # Convert PIL Image to bytes
                            img_buffer = io.BytesIO()
                            image.save(img_buffer, format="PNG")
                            img_bytes = img_buffer.getvalue()

                            # Try to get page number
                            page_no = None
                            if hasattr(item, "prov") and item.prov:
                                page_no = item.prov[0].page_no if item.prov[0].page_no else None

                            figures.append((img_bytes, page_no, fig_index))
                            fig_index += 1
                    except Exception as e:
                        logger.warning(
                            "Failed to extract figure image",
                            figure_index=fig_index,
                            error=str(e),
                        )
                        fig_index += 1

            return markdown_text, figures
        except Exception as docling_err:
            # Second choice, not a silent downgrade: extract the text with a
            # dependency-light parser. Docling is absent by design in the slim
            # deployment image (it carries PyTorch/transformers/timm, ~3GB and
            # several hundred MB resident — more than a 512MB host has), so this
            # is the normal path there rather than an error path.
            #
            # What is lost is stated plainly in parsing/light_parsers.py: tables
            # arrive as cell text rather than structure, and figures are not
            # extracted, so nothing goes for vision captioning. Everything
            # downstream — chunking, retrieval, citations, the critic — is
            # unchanged and simply sees plainer source text.
            from parsing import light_parsers

            if light_parsers.can_parse(ext):
                try:
                    text = light_parsers.extract_text(file_bytes, ext)
                    logger.info(
                        "Parsed with the lightweight extractor (Docling unavailable)",
                        file_extension=ext,
                        chars=len(text),
                    )
                    return text, []
                except Exception as light_err:
                    logger.error(
                        "Lightweight extraction also failed",
                        file_extension=ext,
                        error=str(light_err),
                    )
                    raise RuntimeError(
                        f"Could not parse .{ext} document: {light_err}"
                    ) from light_err

            # Decoding a PDF/DOCX/XLSX as UTF-8 does not "degrade gracefully" — it
            # produces mojibake that chunks, embeds and retrieves perfectly happily
            # while meaning nothing. Ingestion then reports success and the failure
            # only surfaces much later as an unexplained "insufficient evidence".
            #
            # Verified live 2026-09-18: a torch/torchvision ABI mismatch broke
            # Docling's layout model, and a PDF ingested as 173 chunks of binary
            # garbage with no error anywhere. Fail loudly for formats that cannot be
            # text, so the document is marked FAILED and the cause is visible.
            if ext in _BINARY_DOCUMENT_FORMATS:
                logger.error(
                    "Document parsing failed and no text fallback is possible for "
                    "this format — refusing to store undecodable bytes as text",
                    file_extension=ext,
                    error=str(docling_err),
                )
                raise RuntimeError(
                    f"Could not parse .{ext} document: {docling_err}"
                ) from docling_err

            logger.warning(
                "Docling conversion failed, using fallback text decoding",
                file_extension=ext,
                error=str(docling_err),
            )
            text = file_bytes.decode("utf-8", errors="ignore")
            return text, []

    finally:
        # Clean up temp file
        try:
            tmp_path.unlink()
        except OSError:
            pass


async def _caption_image(image_bytes: bytes, page_number: int | None, figure_index: int) -> FigureCaption:
    """Send an extracted image to Gemini 3.5 Flash-Lite for descriptive captioning.

    Uses Gemini 3.5 Flash-Lite because:
    - Cheapest Gemini model ($0.30/1M input tokens)
    - Free tier available
    - Strong vision understanding for charts, diagrams, and photos
    """
    client = _get_gemini_client()

    # Encode image as base64 for the API
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are a document analysis assistant. Describe the content of "
        "this image in detail. If it's a chart, extract the key data points "
        "and trends. If it's a diagram, describe the components and their "
        "relationships. If it's a photo, describe what it depicts. "
        "Be factual and concise. Your description will be used for "
        "semantic search, so include all relevant keywords and data."
    )

    try:
        # Use Gemini's native multimodal input
        from google.genai import types

        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-3.5-flash-lite",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        types.Part.from_text(
                            "Describe this image from a document in detail for search indexing."
                        ),
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=500,
            ),
        )

        caption = response.text or "Image could not be described."
        logger.debug(
            "Figure captioned successfully",
            figure_index=figure_index,
            caption_length=len(caption),
        )

    except Exception as e:
        logger.warning(
            "Vision captioning failed, using fallback",
            figure_index=figure_index,
            error=str(e),
        )
        caption = f"[Image on page {page_number or 'unknown'}, figure {figure_index}]"

    return FigureCaption(
        caption=caption,
        page_number=page_number,
        figure_index=figure_index,
    )


async def parse_document(file_bytes: bytes, filename: str) -> ParseResult:
    """Parse a document into structured Markdown + figure captions.

    This is the main entry point for the multimodal parser. It:
    1. Uses Docling to extract layout-aware text, tables, and detect figures.
    2. Sends each detected figure to Gemini 3.5 Flash-Lite for captioning.
    3. Returns a ParseResult with Markdown, figure captions, AND raw images
       (for native multimodal embedding via Jina CLIP v2).

    Args:
        file_bytes: Raw bytes of the uploaded document.
        filename: Original filename (used to detect format).

    Returns:
        ParseResult with markdown_text, figure_captions, and extracted_images.
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
    logger.info("Starting multimodal parsing", filename=filename, format=ext)

    # Step 1: Run Docling conversion in a thread (it's synchronous / CPU-bound)
    markdown_text, figures = await asyncio.to_thread(
        _docling_convert_sync, file_bytes, filename
    )

    logger.info(
        "Docling extraction complete",
        markdown_length=len(markdown_text),
        figures_found=len(figures),
    )

    # Step 2: Caption all extracted figures concurrently via Gemini
    figure_captions: list[FigureCaption] = []
    if figures:
        caption_tasks = [
            _caption_image(img_bytes, page_no, fig_idx)
            for img_bytes, page_no, fig_idx in figures
        ]
        figure_captions = await asyncio.gather(*caption_tasks)
        logger.info("All figures captioned", count=len(figure_captions))

    return ParseResult(
        markdown_text=markdown_text,
        figure_captions=figure_captions,
        extracted_images=figures,  # Pass raw images for direct embedding
        source_type=ext,
    )
