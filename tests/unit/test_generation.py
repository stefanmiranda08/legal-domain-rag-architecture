"""Tests for the generation module."""

from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

import pytest

from src.generation import (
    build_context,
    format_citations,
    generate_answer,
    GeneratedAnswer,
)
from src.retrieval import RetrievedChunk


class TestBuildContext:
    """Tests for building context from retrieved chunks."""

    def test_builds_context_from_chunks(self):
        """Should format chunks into context string."""
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                text="First chunk content.",
                citation="Case A [2020]",
                score=0.9,
                chunk_index=0,
            ),
            RetrievedChunk(
                chunk_id="c2",
                document_id="d2",
                text="Second chunk content.",
                citation="Case B [2021]",
                score=0.85,
                chunk_index=0,
            ),
        ]

        context = build_context(chunks)

        assert "Case A [2020]" in context
        assert "First chunk content." in context
        assert "Case B [2021]" in context
        assert "Second chunk content." in context

    def test_empty_chunks_returns_empty_context(self):
        """Should return empty string for no chunks."""
        context = build_context([])

        assert context == ""

    def test_includes_citation_for_each_chunk(self):
        """Each chunk should be labeled with its citation."""
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                text="Content here.",
                citation="Smith v Jones [2020]",
                score=0.9,
                chunk_index=0,
            ),
        ]

        context = build_context(chunks)

        assert "Smith v Jones [2020]" in context


class TestFormatCitations:
    """Tests for formatting citation output."""

    def test_formats_citations_from_chunks(self):
        """Should create Citation objects from chunks."""
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                text="First chunk content that is relevant.",
                citation="Case A [2020]",
                score=0.9,
                chunk_index=0,
            ),
        ]

        citations = format_citations(chunks)

        assert len(citations) == 1
        assert citations[0].citation == "Case A [2020]"
        assert citations[0].relevance_score == 0.9

    def test_truncates_long_excerpts(self):
        """Should truncate excerpts that are too long."""
        long_text = "x" * 500
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                text=long_text,
                citation="Case A [2020]",
                score=0.9,
                chunk_index=0,
            ),
        ]

        citations = format_citations(chunks, max_excerpt_length=100)

        assert len(citations[0].excerpt) <= 103  # 100 + "..."

    def test_deduplicates_by_document_id(self):
        """Should not duplicate citations for same document."""
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                text="Chunk 1",
                citation="Case A [2020]",
                score=0.9,
                chunk_index=0,
            ),
            RetrievedChunk(
                chunk_id="c2",
                document_id="d1",  # Same document
                text="Chunk 2",
                citation="Case A [2020]",
                score=0.85,
                chunk_index=1,
            ),
        ]

        citations = format_citations(chunks)

        # Should only have one citation for the document
        assert len(citations) == 1


class TestGeneratedAnswer:
    """Tests for GeneratedAnswer dataclass."""

    def test_answer_creation(self):
        """GeneratedAnswer should store all fields."""
        from src.models import Citation

        answer = GeneratedAnswer(
            text="The court held that...",
            citations=[
                Citation(
                    document_id="d1",
                    citation="Case [2020]",
                    excerpt="...",
                    relevance_score=0.9,
                )
            ],
            tokens_used=500,
        )

        assert answer.text == "The court held that..."
        assert len(answer.citations) == 1
        assert answer.tokens_used == 500


class TestGenerateAnswer:
    """Tests for answer generation."""

    @pytest.fixture
    def mock_openai(self):
        """Create a mock OpenAI client."""
        with patch("src.generation.ChatOpenAI") as mock:
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "This is the generated answer based on the legal documents."
            mock_response.usage_metadata = {"total_tokens": 500}
            mock_llm.invoke.return_value = mock_response
            mock.return_value = mock_llm
            yield mock

    def test_generates_answer_from_chunks(self, mock_openai):
        """Should generate answer using LLM."""
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                text="The standard of care in negligence...",
                citation="Smith v Jones [2020]",
                score=0.9,
                chunk_index=0,
            ),
        ]

        result = generate_answer(
            query="What is the standard of care?",
            chunks=chunks,
            openai_api_key="test-key",
        )

        assert result.text == "This is the generated answer based on the legal documents."
        assert len(result.citations) == 1
        mock_openai.return_value.invoke.assert_called_once()

    def test_handles_empty_chunks(self, mock_openai):
        """Should handle case with no chunks."""
        result = generate_answer(
            query="Some query",
            chunks=[],
            openai_api_key="test-key",
        )

        assert "no relevant documents" in result.text.lower() or result.citations == []

    def test_passes_model_parameter(self, mock_openai):
        """Should use specified model."""
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                text="Content",
                citation="Case [2020]",
                score=0.9,
                chunk_index=0,
            ),
        ]

        generate_answer(
            query="Test query",
            chunks=chunks,
            openai_api_key="test-key",
            model="gpt-4o",
        )

        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs.get("model") == "gpt-4o"
