"""Chunking strategies for processed documents.

Uses LangChain's MarkdownHeaderTextSplitter to chunk text based on structure,
and creates separate chunks for figure captions with full provenance metadata.
"""
import hashlib
from uuid import UUID

from langchain_text_splitters import MarkdownHeaderTextSplitter

from api.schemas.ingestion import Chunk
from parsing.parsers import FigureCaption


def _body_without_headings(text: str) -> str:
    """The chunk's actual content, with its markdown heading lines removed."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ).strip()


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
        contextual_text = (
            f"[{' | '.join(context_parts)}]\n{text}" if context_parts else text
        )

        # Generate deterministic chunk ID
        hash_input = f"{document_id}_{i}_{text}".encode("utf-8")
        chunk_id = hashlib.sha256(hash_input).hexdigest()[:16]

        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=contextual_text,
            section=section,
            source_type=source_type,
            content_type="text",
            parent_element=f"chunk_{i}",
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
