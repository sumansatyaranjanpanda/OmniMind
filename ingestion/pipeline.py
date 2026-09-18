"""End-to-end multimodal document ingestion pipeline.

Orchestrates:
1. Download from MinIO
2. Multimodal parsing via Docling + Gemini Flash-Lite
3. Semantic chunking (text + figure captions)
4. Matryoshka 256-dim embedding via Gemini Embedding 2 (text AND images)
5. Upsert to Pinecone / In-Memory Vector Store
"""

from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.document import Document
from api.storage import download_file
from chunking.strategies import chunk_figures, chunk_markdown
from parsing.parsers import parse_document
from retrieval.pinecone_client import upsert_chunks, upsert_images

logger = structlog.get_logger(__name__)


async def process_document(document_id: UUID, db: AsyncSession) -> None:
    """Orchestrates the ingestion pipeline for a single document.

    Flow:
    1. Fetch document metadata from DB.
    2. Download file from MinIO.
    3. Parse with Docling (text + tables + figures) with plain-text fast path.
    4. Caption figures with Gemini Flash-Lite vision model.
    5. Chunk text and figure captions separately.
    6. Embed text chunks + caption chunks via Gemini Embedding 2 (256-dim).
    7. Embed raw images via Gemini Multimodal Embeddings (256-dim).
    8. Upsert ALL vectors to Pinecone / Vector Store.
    9. Update DB status at each stage.
    """
    logger.info("Starting document processing", document_id=str(document_id))

    # Fetch document
    doc = await db.get(Document, document_id)
    if not doc:
        logger.error("Document not found", document_id=str(document_id))
        return

    try:
        # 1. Download from MinIO
        logger.debug("Downloading file from storage", s3_key=doc.s3_key)
        file_bytes = await download_file(doc.s3_key)

        # 2. Multimodal parsing (Docling + Gemini captioning)
        logger.debug("Parsing document (multimodal)", filename=doc.filename)
        parse_result = await parse_document(file_bytes, doc.filename)
        doc.status = "PARSED"
        await db.commit()

        logger.info(
            "Parse complete",
            markdown_length=len(parse_result.markdown_text),
            figures_found=len(parse_result.figure_captions),
        )

        # 3. Chunk text content
        logger.debug("Chunking markdown text")
        ext = parse_result.source_type
        text_chunks = chunk_markdown(
            parse_result.markdown_text,
            document_id,
            source_type=ext,
            # Names the source inside each chunk so a query mentioning the document's
            # subject can still match passages that never repeat that subject's name.
            document_title=doc.filename,
        )

        # 4. Chunk figure captions (for LLM citation in answers)
        figure_chunks = chunk_figures(
            parse_result.figure_captions,
            document_id,
            source_type=ext,
            text_chunk_count=len(text_chunks),
        )

        # 5. Combine text + caption chunks
        all_text_chunks = text_chunks + figure_chunks
        doc.status = "CHUNKED"
        doc.chunk_count = len(all_text_chunks) + len(parse_result.extracted_images)
        await db.commit()

        logger.info(
            "Chunking complete",
            text_chunks=len(text_chunks),
            figure_chunks=len(figure_chunks),
            total_text_chunks=len(all_text_chunks),
            raw_images=len(parse_result.extracted_images),
        )

        # 6. Embed and upsert text + caption chunks (Gemini Embedding 2)
        tenant_id = str(doc.user_id)
        logger.debug("Embedding text chunks", count=len(all_text_chunks))
        await upsert_chunks(all_text_chunks, tenant_id=tenant_id)

        # 7. Embed and upsert raw images (Gemini Multimodal Embedding 256d)
        if parse_result.extracted_images:
            logger.debug(
                "Embedding images",
                count=len(parse_result.extracted_images),
            )
            await upsert_images(
                images=parse_result.extracted_images,
                document_id=str(document_id),
                source_type=ext,
                tenant_id=tenant_id,
            )

        # 8. Finalize
        doc.status = "EMBEDDED"
        await db.commit()
        logger.info(
            "Document processing complete",
            document_id=str(document_id),
            text_chunks=len(all_text_chunks),
            image_embeddings=len(parse_result.extracted_images),
        )

    except Exception as e:
        logger.exception("Document processing failed", document_id=str(document_id), error=str(e))
        doc.status = "FAILED"
        await db.commit()
        raise
