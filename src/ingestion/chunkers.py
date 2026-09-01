"""Document chunking strategies for the ingestion pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import ChunkingStrategy


@dataclass
class Chunk:
    """A chunk of text from a document."""

    text: str
    index: int
    token_count: int


class BaseChunker(ABC):
    """Base class for chunking strategies."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self._tokenizer.encode(text))

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks."""
        pass


class FixedSizeChunker(BaseChunker):
    """
    Split text into fixed-size chunks based on token count.

    Uses RecursiveCharacterTextSplitter with tiktoken encoding
    to ensure chunks respect token limits while maintaining
    readable boundaries.
    """

    def chunk(self, text: str) -> list[Chunk]:
        if not text or not text.strip():
            return []

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        texts = splitter.split_text(text)

        return [
            Chunk(
                text=t,
                index=i,
                token_count=self.count_tokens(t),
            )
            for i, t in enumerate(texts)
        ]


class ParagraphChunker(BaseChunker):
    """
    Split text on paragraph boundaries, merging small paragraphs
    and splitting large ones.

    Preserves document structure by respecting natural breaks
    in legal documents.
    """

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[Chunk]:
        if not text or not text.strip():
            return []

        # Split on double newlines (paragraph boundaries)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If paragraph alone exceeds max, split it further
            if para_tokens > self.max_chunk_size:
                # Save current chunk if non-empty
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0

                # Split large paragraph using recursive splitter
                sub_chunks = self._split_large_paragraph(para)
                chunks.extend(sub_chunks)
                continue

            # If adding this paragraph exceeds max, save current and start new
            if current_tokens + para_tokens > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = para
                current_tokens = para_tokens
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_tokens += para_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        # Merge chunks that are too small
        merged_chunks = self._merge_small_chunks(chunks)

        return [
            Chunk(
                text=t,
                index=i,
                token_count=self.count_tokens(t),
            )
            for i, t in enumerate(merged_chunks)
        ]

    def _split_large_paragraph(self, text: str) -> list[str]:
        """Split a large paragraph that exceeds max_chunk_size."""
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(text)

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge consecutive chunks that are below min_chunk_size."""
        if not chunks:
            return []

        merged = []
        current = chunks[0]
        current_tokens = self.count_tokens(current)

        for chunk in chunks[1:]:
            chunk_tokens = self.count_tokens(chunk)

            # If current is too small and combined doesn't exceed max, merge
            if current_tokens < self.min_chunk_size:
                combined = current + "\n\n" + chunk
                combined_tokens = self.count_tokens(combined)

                if combined_tokens <= self.max_chunk_size:
                    current = combined
                    current_tokens = combined_tokens
                    continue

            # Save current and start new
            merged.append(current)
            current = chunk
            current_tokens = chunk_tokens

        # Don't forget the last one
        merged.append(current)

        return merged


class RecursiveChunker(BaseChunker):
    """
    Split text using recursive character splitting with
    legal-document-specific separators.

    Prioritizes splitting at section boundaries, numbered paragraphs,
    and natural sentence breaks common in legal documents.
    """

    # Separators optimized for legal documents
    LEGAL_SEPARATORS = [
        "\n\n\n",  # Major section breaks
        "\n\n",  # Paragraph breaks
        "\n",  # Line breaks
        ". ",  # Sentence breaks
        "; ",  # Clause breaks (common in legal text)
        ", ",  # Phrase breaks
        " ",  # Word breaks
        "",  # Character breaks (last resort)
    ]

    def chunk(self, text: str) -> list[Chunk]:
        if not text or not text.strip():
            return []

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.LEGAL_SEPARATORS,
        )

        texts = splitter.split_text(text)

        return [
            Chunk(
                text=t,
                index=i,
                token_count=self.count_tokens(t),
            )
            for i, t in enumerate(texts)
        ]


def get_chunker(
    strategy: ChunkingStrategy,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs,
) -> BaseChunker:
    """
    Factory function to get a chunker for a given strategy.

    Args:
        strategy: The chunking strategy to use.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Number of overlapping tokens between chunks.
        **kwargs: Additional strategy-specific parameters.

    Returns:
        Configured chunker instance.
    """
    if strategy == ChunkingStrategy.FIXED:
        return FixedSizeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif strategy == ChunkingStrategy.PARAGRAPH:
        return ParagraphChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=kwargs.get("min_chunk_size", 100),
            max_chunk_size=kwargs.get("max_chunk_size", 1000),
        )
    elif strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
