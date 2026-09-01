"""Tests for the retrieval module."""

from datetime import date
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from qdrant_client.models import ScoredPoint

from src.models import ChunkingStrategy, DocumentType, Jurisdiction, QueryFilters
from src.retrieval import (
    build_qdrant_filter,
    search_chunks,
    RetrievedChunk,
)


class TestBuildQdrantFilter:
    """Tests for building Qdrant filter from QueryFilters."""

    def test_empty_filters_returns_none(self):
        """Empty filters should return None (no filter)."""
        filters = QueryFilters()
        result = build_qdrant_filter(filters)

        assert result is None

    def test_jurisdiction_filter(self):
        """Should create filter for jurisdiction."""
        filters = QueryFilters(jurisdiction=Jurisdiction.NEW_SOUTH_WALES)
        result = build_qdrant_filter(filters)

        assert result is not None
        assert len(result.must) == 1
        assert result.must[0].key == "jurisdiction"

    def test_document_type_filter(self):
        """Should create filter for document type."""
        filters = QueryFilters(document_type=DocumentType.DECISION)
        result = build_qdrant_filter(filters)

        assert result is not None
        assert len(result.must) == 1
        assert result.must[0].key == "document_type"

    def test_date_range_filter(self):
        """Should create filter for date range."""
        filters = QueryFilters(
            date_from=date(2020, 1, 1),
            date_to=date(2023, 12, 31),
        )
        result = build_qdrant_filter(filters)

        assert result is not None
        # Should have two conditions (gte and lte)
        assert len(result.must) == 2

    def test_combined_filters(self):
        """Should combine multiple filters."""
        filters = QueryFilters(
            jurisdiction=Jurisdiction.COMMONWEALTH,
            document_type=DocumentType.PRIMARY_LEGISLATION,
            date_from=date(2020, 1, 1),
        )
        result = build_qdrant_filter(filters)

        assert result is not None
        assert len(result.must) == 3


class TestRetrievedChunk:
    """Tests for RetrievedChunk dataclass."""

    def test_chunk_creation(self):
        """RetrievedChunk should store all fields."""
        chunk = RetrievedChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            text="Some legal text...",
            citation="Smith v Jones [2020]",
            score=0.89,
            chunk_index=0,
        )

        assert chunk.document_id == "doc_1"
        assert chunk.score == 0.89


class TestSearchChunks:
    """Tests for vector search functionality."""

    @pytest.fixture
    def mock_qdrant(self):
        """Create a mock Qdrant client."""
        return Mock()

    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        with patch("src.retrieval.OpenAI") as mock:
            mock_client = Mock()
            mock_embedding = Mock()
            mock_embedding.embedding = [0.1] * 1536
            mock_response = Mock()
            mock_response.data = [mock_embedding]
            mock_client.embeddings.create.return_value = mock_response
            mock.return_value = mock_client
            yield mock

    def test_search_returns_chunks(self, mock_qdrant, mock_openai):
        """Should return retrieved chunks from search."""
        # Mock Qdrant search result
        mock_point = Mock(spec=ScoredPoint)
        mock_point.id = str(uuid4())
        mock_point.score = 0.85
        mock_point.payload = {
            "document_id": str(uuid4()),
            "chunk_index": 0,
            "citation": "Test Case [2020]",
        }

        mock_result = Mock()
        mock_result.points = [mock_point]
        mock_qdrant.query_points.return_value = mock_result

        # Mock chunk text retrieval
        with patch("src.retrieval.get_chunk_texts") as mock_get_texts:
            mock_get_texts.return_value = {mock_point.id: "This is the chunk text."}

            chunks = search_chunks(
                query="test query",
                qdrant_client=mock_qdrant,
                strategy=ChunkingStrategy.RECURSIVE,
                openai_api_key="test-key",
                top_k=5,
            )

        assert len(chunks) == 1
        assert chunks[0].score == 0.85
        assert chunks[0].citation == "Test Case [2020]"

    def test_search_with_filters(self, mock_qdrant, mock_openai):
        """Should apply filters to search."""
        mock_result = Mock()
        mock_result.points = []
        mock_qdrant.query_points.return_value = mock_result

        with patch("src.retrieval.get_chunk_texts") as mock_get_texts:
            mock_get_texts.return_value = {}

            filters = QueryFilters(jurisdiction=Jurisdiction.NEW_SOUTH_WALES)

            search_chunks(
                query="test query",
                qdrant_client=mock_qdrant,
                strategy=ChunkingStrategy.RECURSIVE,
                openai_api_key="test-key",
                filters=filters,
            )

        # Verify filter was passed to Qdrant
        call_kwargs = mock_qdrant.query_points.call_args[1]
        assert call_kwargs.get("query_filter") is not None

    def test_search_respects_top_k(self, mock_qdrant, mock_openai):
        """Should limit results to top_k."""
        mock_result = Mock()
        mock_result.points = []
        mock_qdrant.query_points.return_value = mock_result

        with patch("src.retrieval.get_chunk_texts") as mock_get_texts:
            mock_get_texts.return_value = {}

            search_chunks(
                query="test query",
                qdrant_client=mock_qdrant,
                strategy=ChunkingStrategy.RECURSIVE,
                openai_api_key="test-key",
                top_k=20,
            )

        call_kwargs = mock_qdrant.query_points.call_args[1]
        assert call_kwargs.get("limit") == 20

    def test_search_empty_query(self, mock_qdrant, mock_openai):
        """Should handle empty query gracefully."""
        with patch("src.retrieval.get_chunk_texts"):
            mock_result = Mock()
            mock_result.points = []
            mock_qdrant.query_points.return_value = mock_result

            chunks = search_chunks(
                query="",
                qdrant_client=mock_qdrant,
                strategy=ChunkingStrategy.RECURSIVE,
                openai_api_key="test-key",
            )

        assert chunks == []
