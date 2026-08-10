"""Pydantic models for the FastAPI layer."""

from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    query: str = Field(..., min_length=1, max_length=500, description="User question")
    k: int = Field(default=10, ge=1, le=50, description="Candidate chunks per retriever")
    top_n: int = Field(default=5, ge=1, le=10, description="Final reranked chunks sent to the LLM")


class SourceChunk(BaseModel):
    """A single retrieved source chunk with provenance metadata."""

    content: str
    page: Optional[int] = None
    source: Optional[str] = None


class Metrics(BaseModel):
    num_chunks_indexed: int
    query_rewrite_ms: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class AskResponse(BaseModel):
    """Structured response from the RAG pipeline."""

    query: str
    rewritten_query: str
    answer: str
    sources: list[SourceChunk]
    metrics: Metrics
