"""Chunking strategies for processed documents.

Splits on markdown structure first, then by size, and creates separate chunks for
figure captions with full provenance metadata.

The size pass is not a refinement — it is load-bearing. Structure-only splitting
assumes the document HAS structure, and silently produces one chunk containing the
entire document when it doesn't: a prose PDF, a scanned report, anything Docling
renders without `#` headings. Confirmed 2026-09-18 against a real PDF, which parsed
to a single 211KB chunk. That breaks three things at once — the embedding model
truncates at its token limit so most of the document is never represented, retrieval
precision collapses because the only retrievable unit is "the whole document", and
Pinecone rejects the upsert outright (metadata capped at 40,960 bytes per vector),
so ingestion fails rather than degrading.
"""
import hashlib
from uuid import UUID

from chunking.splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from api.schemas.ingestion import Chunk
from parsing.parsers import FigureCaption

# Sized to sit comfortably under both hard limits it has to respect: the embedding
# model's input window, and Pinecone's 40,960-byte metadata ceiling (the chunk text
# travels in metadata so it can be returned with a citation). Small enough that a
# retrieved passage is precise rather than a page of mixed topics; the overlap keeps
# a fact that straddles a boundary from being cut in half.
MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP_CHARS = 200


def _body_without_headings(text: str) -> str:
    """The chunk's actual content, with its markdown heading lines removed."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ).strip()


def _split_oversized(text: str) -> list[str]:
    """Split on natural boundaries — paragraph, line, sentence, word — largest first.

    Returns a single-element list when the text already fits, so the common case of a
    well-structured document is untouched and keeps its section-aligned chunks.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [piece for piece in splitter.split_text(text) if piece.strip()]


def chunk_markdown(
    markdown_text: str,
    document_id: UUID,
    source_type: str,
    document_title: str | None = None,
) -> list[Chunk]:
    """Splits markdown text by headers and returns Chunk objects.

    Two things happen here beyond the split, both aimed at retrieval quality:

    **Heading-only chunks are dropped.** When two headings sit at the same level with
    nothing between them, the splitter emits the first as a chunk containing only its own
    title — a real résumé produced `"## Work Experience"` (18 characters) and
    `"## Education"` (12) as standalone entries. They can never answer a question, but they
    still occupy top-k slots and dilute the ranking, pushing real evidence out of the
    results.

    **Each chunk is prefixed with where it came from.** A chunk sliced out of a document
    loses the context that made it meaningful: the passage listing someone's jobs starts
    "## Radical Minds Technologies" and never repeats their name, so a search for that
    person scores the contact-details header above their actual work history. Naming the
    document and section inside the chunk restores what the split removed, and it helps at
    all three stages — dense embedding, BM25 keyword matching, and cross-encoder reranking
    all read this text.
    """
    # Define the headers we want to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    docs = markdown_splitter.split_text(markdown_text)

    chunks = []
    for i, doc in enumerate(docs):
        text = doc.page_content
        metadata = doc.metadata

        # A heading with no body under it is a label, not evidence.
        if not _body_without_headings(text):
            continue

        # Build section hierarchy string (e.g. "Header 1 > Header 2")
        section_parts = []
        for key in ["Header 1", "Header 2", "Header 3"]:
            if key in metadata:
                section_parts.append(metadata[key])

        section = " > ".join(section_parts) if section_parts else "Document Root"

        context_parts = []
        if document_title:
            context_parts.append(document_title)
        if section and section != "Document Root":
            context_parts.append(section)
        prefix = f"[{' | '.join(context_parts)}]\n" if context_parts else ""

        # Every piece carries the document/section prefix, not just the first. A
        # sub-chunk from the middle of a section is exactly the case that loses its
        # subject otherwise — which is the problem the prefix exists to solve.
        for j, piece in enumerate(_split_oversized(text)):
            contextual_text = f"{prefix}{piece}"

            # Deterministic, and distinct per sub-piece: keying the hash on `i` alone
            # would give every piece of one section the same id, and each upsert would
            # overwrite the last — silently keeping only the final piece.
            hash_input = f"{document_id}_{i}_{j}_{piece}".encode("utf-8")
            chunk_id = hashlib.sha256(hash_input).hexdigest()[:16]

            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=contextual_text,
                section=section,
                source_type=source_type,
                content_type="text",
                parent_element=f"chunk_{i}_{j}",
            )
            chunks.append(chunk)

    return chunks


def chunk_figures(
    figure_captions: list[FigureCaption],
    document_id: UUID,
    source_type: str,
    text_chunk_count: int = 0,
) -> list[Chunk]:
    """Creates Chunk objects from extracted figure captions.
    
    Each figure caption becomes its own chunk with content_type="figure_caption",
    preserving page number and figure index as provenance.
    
    Args:
        figure_captions: List of FigureCaption objects from the parser.
        document_id: ID of the parent document.
        source_type: Extension or type of the source file.
        text_chunk_count: Number of text chunks already created (for unique indexing).
        
    Returns:
        List of Chunk objects for figure captions.
    """
    chunks = []
    for j, fig in enumerate(figure_captions):
        global_index = text_chunk_count + j
        
        # Generate deterministic chunk ID
        hash_input = f"{document_id}_fig_{j}_{fig.caption}".encode("utf-8")
        chunk_id = hashlib.sha256(hash_input).hexdigest()[:16]
        
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=fig.caption,
            page=fig.page_number,
            section=f"Figure {fig.figure_index + 1}",
            source_type=source_type,
            content_type=fig.content_type,
            parent_element=f"figure_{fig.figure_index}",
        )
        chunks.append(chunk)
        
    return chunks
