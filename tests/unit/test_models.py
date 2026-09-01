"""Tests for data models."""

from datetime import date, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.models import (
    ChunkingStrategy,
    DocumentType,
    Jurisdiction,
    QueryFilters,
    QueryRequest,
    QueryResponse,
    Citation,
    DocumentMetadata,
)


class TestEnums:
    """Tests for enum types."""

    def test_document_type_values(self):
        """DocumentType should have all expected values."""
        assert DocumentType.PRIMARY_LEGISLATION.value == "primary_legislation"
        assert DocumentType.SECONDARY_LEGISLATION.value == "secondary_legislation"
        assert DocumentType.BILL.value == "bill"
        assert DocumentType.DECISION.value == "decision"

    def test_jurisdiction_values(self):
        """Jurisdiction should have all expected values."""
        assert Jurisdiction.COMMONWEALTH.value == "commonwealth"
        assert Jurisdiction.NEW_SOUTH_WALES.value == "new_south_wales"
        assert Jurisdiction.QUEENSLAND.value == "queensland"
        assert Jurisdiction.WESTERN_AUSTRALIA.value == "western_australia"
        assert Jurisdiction.SOUTH_AUSTRALIA.value == "south_australia"
        assert Jurisdiction.TASMANIA.value == "tasmania"
        assert Jurisdiction.NORFOLK_ISLAND.value == "norfolk_island"

    def test_chunking_strategy_values(self):
        """ChunkingStrategy should have all expected values."""
        assert ChunkingStrategy.FIXED.value == "fixed"
        assert ChunkingStrategy.PARAGRAPH.value == "paragraph"
        assert ChunkingStrategy.RECURSIVE.value == "recursive"


class TestQueryFilters:
    """Tests for QueryFilters schema."""

    def test_empty_filters_valid(self):
        """Empty filters should be valid."""
        filters = QueryFilters()

        assert filters.jurisdiction is None
        assert filters.document_type is None
        assert filters.date_from is None
        assert filters.date_to is None

    def test_filters_with_all_fields(self):
        """Filters with all fields should be valid."""
        filters = QueryFilters(
            jurisdiction=Jurisdiction.NEW_SOUTH_WALES,
            document_type=DocumentType.DECISION,
            date_from=date(2020, 1, 1),
            date_to=date(2023, 12, 31),
        )

        assert filters.jurisdiction == Jurisdiction.NEW_SOUTH_WALES
        assert filters.document_type == DocumentType.DECISION
        assert filters.date_from == date(2020, 1, 1)
        assert filters.date_to == date(2023, 12, 31)

    def test_filters_date_from_string(self):
        """Filters should accept date as string."""
        filters = QueryFilters(date_from="2020-01-01")

        assert filters.date_from == date(2020, 1, 1)


class TestQueryRequest:
    """Tests for QueryRequest schema."""

    def test_minimal_request(self):
        """Request with only query should be valid."""
        request = QueryRequest(query="What is negligence?")

        assert request.query == "What is negligence?"
        assert request.filters is None
        assert request.chunking_strategy == ChunkingStrategy.RECURSIVE
        assert request.top_k == 10

    def test_full_request(self):
        """Request with all fields should be valid."""
        request = QueryRequest(
            query="What is negligence?",
            filters=QueryFilters(jurisdiction=Jurisdiction.COMMONWEALTH),
            chunking_strategy=ChunkingStrategy.FIXED,
            top_k=5,
        )

        assert request.query == "What is negligence?"
        assert request.filters.jurisdiction == Jurisdiction.COMMONWEALTH
        assert request.chunking_strategy == ChunkingStrategy.FIXED
        assert request.top_k == 5

    def test_empty_query_invalid(self):
        """Empty query should be invalid."""
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_top_k_must_be_positive(self):
        """top_k must be positive."""
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)


class TestCitation:
    """Tests for Citation schema."""

    def test_citation_creation(self):
        """Citation should be created with all fields."""
        citation = Citation(
            document_id=str(uuid4()),
            citation="Smith v Jones [2020] NSWSC 123",
            excerpt="The court held that...",
            document_date=date(2020, 5, 15),
            relevance_score=0.85,
        )

        assert citation.citation == "Smith v Jones [2020] NSWSC 123"
        assert citation.relevance_score == 0.85
        assert citation.document_date == date(2020, 5, 15)

    def test_citation_date_optional(self):
        """Citation date should be optional."""
        citation = Citation(
            document_id=str(uuid4()),
            citation="Some Act 2020",
            excerpt="Section 1 states...",
            relevance_score=0.9,
        )

        assert citation.document_date is None


class TestQueryResponse:
    """Tests for QueryResponse schema."""

    def test_response_creation(self):
        """QueryResponse should be created with all fields."""
        response = QueryResponse(
            answer="Negligence requires...",
            citations=[
                Citation(
                    document_id=str(uuid4()),
                    citation="Smith v Jones [2020]",
                    excerpt="...",
                    relevance_score=0.9,
                )
            ],
            query_id="q_123",
            latency_ms=150,
            chunks_retrieved=10,
            tokens_used=500,
        )

        assert response.answer == "Negligence requires..."
        assert len(response.citations) == 1
        assert response.latency_ms == 150


class TestDocumentMetadata:
    """Tests for DocumentMetadata schema."""

    def test_document_metadata_creation(self):
        """DocumentMetadata should be created from corpus data."""
        metadata = DocumentMetadata(
            version_id="doc_123",
            citation="Smith v Jones [2020] NSWSC 456",
            document_type=DocumentType.DECISION,
            jurisdiction=Jurisdiction.NEW_SOUTH_WALES,
            source="nsw_caselaw",
            document_date=date(2020, 5, 15),
            url="https://example.com/doc",
        )

        assert metadata.version_id == "doc_123"
        assert metadata.document_type == DocumentType.DECISION
        assert metadata.document_date == date(2020, 5, 15)

    def test_document_metadata_date_optional(self):
        """DocumentMetadata date should be optional."""
        metadata = DocumentMetadata(
            version_id="doc_123",
            citation="Some Act",
            document_type=DocumentType.PRIMARY_LEGISLATION,
            jurisdiction=Jurisdiction.COMMONWEALTH,
            source="federal_register",
        )

        assert metadata.document_date is None
        assert metadata.url is None
