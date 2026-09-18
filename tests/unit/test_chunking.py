import uuid
from uuid import uuid4

from chunking.strategies import chunk_figures, chunk_markdown
from parsing.parsers import FigureCaption


def test_chunk_markdown_structure():
    """Verify that headers are respected and provenance is attached."""
    markdown_text = (
        "# Title\n\n"
        "Some introductory text.\n\n"
        "## Section 1\n\n"
        "Details for section 1.\n\n"
        "### Subsection A\n\n"
        "Sub-details."
    )
    doc_id = uuid.uuid4()
    chunks = chunk_markdown(markdown_text, doc_id, "md")
    
    # We should have 3 chunks (since text is split into the blocks under headers)
    assert len(chunks) == 3
    
    # Check the first chunk
    assert chunks[0].section == "Title"
    assert "Some introductory text." in chunks[0].text
    assert "# Title" in chunks[0].text
    assert chunks[0].document_id == doc_id
    assert chunks[0].content_type == "text"
    
    # Check the second chunk
    assert chunks[1].section == "Title > Section 1"
    assert "Details for section 1." in chunks[1].text
    assert "## Section 1" in chunks[1].text
    assert chunks[1].content_type == "text"
    
    # Check the third chunk
    assert chunks[2].section == "Title > Section 1 > Subsection A"
    assert "Sub-details." in chunks[2].text
    assert "### Subsection A" in chunks[2].text
    assert chunks[2].content_type == "text"
    
    # Ensure chunk IDs are deterministic
    chunks_again = chunk_markdown(markdown_text, doc_id, "md")
    assert chunks[0].chunk_id == chunks_again[0].chunk_id


def test_chunk_figures():
    """Verify that figure captions are properly converted to Chunk objects."""
    doc_id = uuid.uuid4()
    
    captions = [
        FigureCaption(
            caption="A bar chart showing revenue growth from $1M to $5M over 2020-2024.",
            page_number=3,
            figure_index=0,
        ),
        FigureCaption(
            caption="Architecture diagram showing API → Parser → Chunker → Pinecone flow.",
            page_number=7,
            figure_index=1,
        ),
    ]
    
    chunks = chunk_figures(captions, doc_id, source_type="pdf", text_chunk_count=10)
    
    assert len(chunks) == 2
    
    # First figure chunk
    assert chunks[0].content_type == "figure_caption"
    assert "revenue growth" in chunks[0].text
    assert chunks[0].page == 3
    assert chunks[0].section == "Figure 1"
    assert chunks[0].document_id == doc_id
    assert chunks[0].source_type == "pdf"
    
    # Second figure chunk
    assert chunks[1].content_type == "figure_caption"
    assert "Architecture diagram" in chunks[1].text
    assert chunks[1].page == 7
    assert chunks[1].section == "Figure 2"
    
    # Ensure chunk IDs are deterministic
    chunks_again = chunk_figures(captions, doc_id, source_type="pdf", text_chunk_count=10)
    assert chunks[0].chunk_id == chunks_again[0].chunk_id


def test_chunk_figures_empty():
    """Verify that empty figure list produces no chunks."""
    doc_id = uuid.uuid4()
    chunks = chunk_figures([], doc_id, source_type="pdf")
    assert chunks == []


# ── Size-based splitting (added 2026-09-18) ─────────────────────
#
# Structure-only splitting assumes the document has structure. A real PDF with no
# markdown headings parsed to a SINGLE 211KB chunk, which the embedding model would
# truncate, retrieval could not use precisely, and Pinecone rejected outright
# (metadata capped at 40,960 bytes per vector), failing the whole ingest.


def test_a_document_with_no_headings_is_still_split():
    from chunking.strategies import MAX_CHUNK_CHARS, chunk_markdown

    prose = ("This sentence is about distributed systems and retrieval. " * 400)
    assert len(prose) > MAX_CHUNK_CHARS * 5

    chunks = chunk_markdown(prose, document_id=uuid4(), source_type="pdf")

    assert len(chunks) > 1, "a heading-less document collapsed into one chunk"
    for c in chunks:
        assert len(c.text) <= MAX_CHUNK_CHARS + 400  # + room for the context prefix


def test_every_sub_chunk_gets_a_unique_id():
    """Hashing on the section index alone made sub-chunks collide, so each upsert
    overwrote the previous one and only the final piece survived."""
    from chunking.strategies import chunk_markdown

    prose = "## Section\n\n" + ("Retrieval augmented generation content. " * 400)
    chunks = chunk_markdown(prose, document_id=uuid4(), source_type="pdf")

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "sub-chunks share an id and would overwrite"


def test_every_sub_chunk_keeps_its_document_and_section_context():
    """A middle sub-chunk is exactly the passage that loses its subject otherwise."""
    from chunking.strategies import chunk_markdown

    prose = "## Work Experience\n\n" + ("Built distributed pipelines at scale. " * 400)
    chunks = chunk_markdown(
        prose, document_id=uuid4(), source_type="pdf", document_title="Resume.pdf"
    )

    assert len(chunks) > 1
    for c in chunks:
        assert "Resume.pdf" in c.text
        assert c.section == "Work Experience"


def test_a_well_structured_document_is_not_over_split():
    """The size pass must not fragment documents that were already fine."""
    from chunking.strategies import chunk_markdown

    md = "## Alpha\n\nShort body one.\n\n## Beta\n\nShort body two.\n"
    chunks = chunk_markdown(md, document_id=uuid4(), source_type="md")

    assert len(chunks) == 2
    assert {c.section for c in chunks} == {"Alpha", "Beta"}
