"""Pydantic schemas for the ingestion and chunking pipeline."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    """Response model for a Document."""
    id: UUID
    filename: str
    status: str
    chunk_count: int = 0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Chunk(BaseModel):
    """A semantic piece of a document, ready for embedding.
    
    Must carry provenance data to trace an answer back to the original source,
    as mandated by the data contract in claude.md.
    """
    chunk_id: str = Field(..., description="Unique ID for this chunk (e.g., hash or doc_id + index)")
    document_id: UUID = Field(..., description="ID of the parent Document")
    text: str = Field(..., description="The actual text content of the chunk")
    
    # Provenance metadata
    page: Optional[int] = Field(None, description="Page number if applicable")
    section: Optional[str] = Field(None, description="Markdown header or section title")
    source_type: str = Field(..., description="e.g., pdf, docx, html")
    content_type: str = Field(
        "text",
        description="Type of content: 'text', 'table', or 'figure_caption'",
    )
    parent_element: Optional[str] = Field(None, description="Path or context of the parent element")
    
    model_config = ConfigDict(from_attributes=True)
