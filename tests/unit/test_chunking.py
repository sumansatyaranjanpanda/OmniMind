import uuid

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
