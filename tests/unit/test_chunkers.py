"""Tests for document chunking strategies."""

import pytest

from src.ingestion.chunkers import (
    Chunk,
    FixedSizeChunker,
    ParagraphChunker,
    RecursiveChunker,
    get_chunker,
)
from src.models import ChunkingStrategy

# Sample legal text for testing
SAMPLE_LEGAL_TEXT = """
REASONS FOR JUDGMENT

Introduction

1. This matter concerns an application for judicial review of a decision made by the Administrative Appeals Tribunal.

2. The applicant seeks to challenge the Tribunal's decision on the grounds that it failed to properly consider relevant evidence.

Background

3. On 15 March 2020, the applicant lodged an application with the relevant government department.

4. The department rejected the application on 1 June 2020, citing insufficient documentation.

5. The applicant appealed this decision to the Tribunal, which affirmed the department's decision on 15 September 2020.

Legal Framework

6. Section 5 of the Administrative Decisions (Judicial Review) Act 1977 (Cth) provides that a person aggrieved by a decision may apply for judicial review.

7. The grounds for review include that the decision was made in breach of the rules of natural justice, or that the decision was not authorised by the enactment.

Consideration

8. The central issue in this case is whether the Tribunal properly considered all relevant material before it.

9. Having reviewed the transcript of the proceedings and the Tribunal's written reasons, I am satisfied that the Tribunal did consider the relevant evidence.

10. The applicant's submission that certain documents were overlooked is not supported by the record.

Conclusion

11. For these reasons, the application for judicial review is dismissed.

12. The applicant is to pay the respondent's costs of the proceeding.
""".strip()


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_chunk_creation(self):
        """Chunk should store text and index."""
        chunk = Chunk(text="Some text", index=0, token_count=5)

        assert chunk.text == "Some text"
        assert chunk.index == 0
        assert chunk.token_count == 5


class TestFixedSizeChunker:
    """Tests for fixed-size chunking strategy."""

    def test_chunks_text_into_fixed_sizes(self):
        """Should split text into approximately equal token-sized chunks."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(SAMPLE_LEGAL_TEXT)

        assert len(chunks) > 1
        # Each chunk should have an index
        assert all(c.index == i for i, c in enumerate(chunks))

    def test_respects_chunk_size(self):
        """Chunks should not exceed chunk_size by much."""
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(SAMPLE_LEGAL_TEXT)

        # Allow some tolerance since tiktoken may not split perfectly
        for chunk in chunks[:-1]:  # Last chunk may be smaller
            assert chunk.token_count <= 60  # chunk_size + tolerance

    def test_handles_empty_text(self):
        """Should return empty list for empty text."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("")

        assert chunks == []

    def test_handles_short_text(self):
        """Short text should produce single chunk."""
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk("This is a short sentence.")

        assert len(chunks) == 1

    def test_overlap_produces_more_chunks(self):
        """Higher overlap should produce more chunks."""
        chunker_no_overlap = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        chunker_with_overlap = FixedSizeChunker(chunk_size=100, chunk_overlap=30)

        chunks_no_overlap = chunker_no_overlap.chunk(SAMPLE_LEGAL_TEXT)
        chunks_with_overlap = chunker_with_overlap.chunk(SAMPLE_LEGAL_TEXT)

        assert len(chunks_with_overlap) >= len(chunks_no_overlap)


class TestParagraphChunker:
    """Tests for paragraph-based chunking strategy."""

    def test_splits_on_paragraph_boundaries(self):
        """Should split text on paragraph boundaries."""
        chunker = ParagraphChunker(min_chunk_size=50, max_chunk_size=200)
        chunks = chunker.chunk(SAMPLE_LEGAL_TEXT)

        assert len(chunks) > 1
        # Chunks should not start/end mid-sentence (approximately)

    def test_merges_small_paragraphs(self):
        """Small paragraphs should be merged together."""
        short_paragraphs = "First.\n\nSecond.\n\nThird.\n\nFourth."
        chunker = ParagraphChunker(min_chunk_size=20, max_chunk_size=100)
        chunks = chunker.chunk(short_paragraphs)

        # Small paragraphs should be merged
        assert len(chunks) < 4

    def test_splits_large_paragraphs(self):
        """Large paragraphs exceeding max_chunk_size should be split."""
        large_paragraph = "This is a sentence. " * 100  # Very long paragraph
        chunker = ParagraphChunker(min_chunk_size=50, max_chunk_size=200)
        chunks = chunker.chunk(large_paragraph)

        assert len(chunks) > 1

    def test_handles_empty_text(self):
        """Should return empty list for empty text."""
        chunker = ParagraphChunker(min_chunk_size=50, max_chunk_size=200)
        chunks = chunker.chunk("")

        assert chunks == []

    def test_preserves_section_headers(self):
        """Section headers should ideally be kept with following content."""
        chunker = ParagraphChunker(min_chunk_size=50, max_chunk_size=500)
        chunks = chunker.chunk(SAMPLE_LEGAL_TEXT)

        # At least some chunks should contain headers
        header_chunks = [c for c in chunks if "Introduction" in c.text or "Background" in c.text]
        assert len(header_chunks) > 0


class TestRecursiveChunker:
    """Tests for recursive character chunking strategy."""

    def test_splits_recursively(self):
        """Should split text using recursive character splitting."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk(SAMPLE_LEGAL_TEXT)

        assert len(chunks) > 1
        assert all(c.index == i for i, c in enumerate(chunks))

    def test_uses_legal_separators(self):
        """Should use legal-document-specific separators."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(SAMPLE_LEGAL_TEXT)

        # Chunks should tend to break at section boundaries
        # This is a soft check - just verify we get reasonable chunks
        assert all(len(c.text.strip()) > 0 for c in chunks)

    def test_handles_empty_text(self):
        """Should return empty list for empty text."""
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=40)
        chunks = chunker.chunk("")

        assert chunks == []

    def test_respects_chunk_size(self):
        """Chunks should approximately respect chunk size."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(SAMPLE_LEGAL_TEXT)

        # Most chunks should be close to chunk_size
        for chunk in chunks[:-1]:
            assert chunk.token_count <= 120  # Allow some tolerance


class TestGetChunker:
    """Tests for chunker factory function."""

    def test_returns_fixed_chunker(self):
        """Should return FixedSizeChunker for FIXED strategy."""
        chunker = get_chunker(ChunkingStrategy.FIXED)
        assert isinstance(chunker, FixedSizeChunker)

    def test_returns_paragraph_chunker(self):
        """Should return ParagraphChunker for PARAGRAPH strategy."""
        chunker = get_chunker(ChunkingStrategy.PARAGRAPH)
        assert isinstance(chunker, ParagraphChunker)

    def test_returns_recursive_chunker(self):
        """Should return RecursiveChunker for RECURSIVE strategy."""
        chunker = get_chunker(ChunkingStrategy.RECURSIVE)
        assert isinstance(chunker, RecursiveChunker)

    def test_passes_custom_parameters(self):
        """Should pass custom parameters to chunker."""
        chunker = get_chunker(
            ChunkingStrategy.FIXED,
            chunk_size=256,
            chunk_overlap=32,
        )
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 32
