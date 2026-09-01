"""Data models: Pydantic schemas and SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Date, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, relationship


# === Enums ===


class DocumentType(str, Enum):
    """Legal document types from the Australian corpus."""

    PRIMARY_LEGISLATION = "primary_legislation"
    SECONDARY_LEGISLATION = "secondary_legislation"
    BILL = "bill"
    DECISION = "decision"


class Jurisdiction(str, Enum):
    """Australian jurisdictions."""

    COMMONWEALTH = "commonwealth"
    NEW_SOUTH_WALES = "new_south_wales"
    QUEENSLAND = "queensland"
    WESTERN_AUSTRALIA = "western_australia"
    SOUTH_AUSTRALIA = "south_australia"
    TASMANIA = "tasmania"
    NORFOLK_ISLAND = "norfolk_island"


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"


# === Pydantic Schemas (API Request/Response) ===


class QueryFilters(BaseModel):
    """Filters for narrowing search results."""

    jurisdiction: Optional[Jurisdiction] = None
    document_type: Optional[DocumentType] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""

    query: str = Field(..., min_length=1)
    filters: Optional[QueryFilters] = None
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    top_k: int = Field(default=10, gt=0)

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class Citation(BaseModel):
    """A citation to a source document."""

    document_id: str
    citation: str
    excerpt: str
    document_date: date | None = None
    relevance_score: float


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""

    answer: str
    citations: list[Citation]
    query_id: str
    latency_ms: int
    chunks_retrieved: int
    tokens_used: int


class DocumentMetadata(BaseModel):
    """Metadata for a document from the corpus."""

    version_id: str
    citation: str
    document_type: DocumentType
    jurisdiction: Jurisdiction
    source: str
    document_date: date | None = None
    url: str | None = None


class IngestRequest(BaseModel):
    """Request body for /ingest endpoint."""

    chunking_strategies: list[ChunkingStrategy] = Field(
        default=[ChunkingStrategy.FIXED, ChunkingStrategy.PARAGRAPH, ChunkingStrategy.RECURSIVE]
    )
    limit: Optional[int] = Field(default=None, gt=0)
    jurisdiction_filter: Optional[Jurisdiction] = None


class IngestResponse(BaseModel):
    """Response body for /ingest endpoint."""

    job_id: str
    status: str
    estimated_documents: int


class IngestStatus(BaseModel):
    """Status response for /ingest/{job_id} endpoint."""

    job_id: str
    status: str
    documents_processed: int
    documents_total: int
    errors: int
    started_at: datetime
    completed_at: Optional[datetime] = None


class EvaluationRequest(BaseModel):
    """Request body for /evaluate endpoint."""

    chunking_strategy: ChunkingStrategy
    test_set_id: str = "default"


class EvaluationMetrics(BaseModel):
    """Aggregated evaluation metrics."""

    recall_at_5: float
    recall_at_10: float
    mrr: float
    avg_latency_ms: float


class PerQueryResult(BaseModel):
    """Per-query evaluation result."""

    query_id: str
    query: str
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    recall_at_5: float


class EvaluationResponse(BaseModel):
    """Response body for /evaluate endpoint."""

    evaluation_id: str
    strategy: ChunkingStrategy
    metrics: EvaluationMetrics
    per_query_results: list[PerQueryResult]


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    qdrant: str
    postgres: str
    timestamp: datetime


# === SQLAlchemy ORM Models ===


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


class Document(Base):
    """Source document metadata."""

    __tablename__ = "documents"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    version_id = Column(String(255), unique=True, nullable=False)
    citation = Column(Text, nullable=False)
    document_type = Column(String(50), nullable=False)
    jurisdiction = Column(String(50), nullable=False)
    source = Column(String(100), nullable=False)
    date = Column(Date, nullable=True)
    url = Column(Text, nullable=True)
    text_length = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document")


class Chunk(Base):
    """Document chunk with embedding reference."""

    __tablename__ = "chunks"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(PGUUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunking_strategy = Column(String(50), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    qdrant_point_id = Column(PGUUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


class QueryLog(Base):
    """Log of user queries for observability."""

    __tablename__ = "query_logs"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    query_text = Column(Text, nullable=False)
    filters = Column(JSON, nullable=True)
    chunking_strategy = Column(String(50), nullable=True)
    chunks_retrieved = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    answer_text = Column(Text, nullable=True)
    citations = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationRun(Base):
    """Results of an evaluation run."""

    __tablename__ = "evaluation_runs"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    chunking_strategy = Column(String(50), nullable=False)
    test_set_id = Column(String(100), nullable=False)
    recall_at_5 = Column(Float, nullable=True)
    recall_at_10 = Column(Float, nullable=True)
    mrr = Column(Float, nullable=True)
    avg_latency_ms = Column(Float, nullable=True)
    per_query_results = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class IngestionJob(Base):
    """Tracking for ingestion jobs."""

    __tablename__ = "ingestion_jobs"

    id = Column(String(100), primary_key=True)
    status = Column(String(50), nullable=False)
    chunking_strategies = Column(JSON, nullable=True)
    documents_total = Column(Integer, nullable=True)
    documents_processed = Column(Integer, default=0)
    errors = Column(Integer, default=0)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class TestQuery(Base):
    """Evaluation test queries."""

    __tablename__ = "test_queries"

    id = Column(String(100), primary_key=True)
    test_set_id = Column(String(100), nullable=False)
    query_text = Column(Text, nullable=False)
    expected_document_ids = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
