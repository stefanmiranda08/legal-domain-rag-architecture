"""
Section reconstruction module for retrieving complete semantic units.

This module provides functionality to reconstruct complete sections from
linked chunks during retrieval, using LLM-based boundary detection.
"""

import logging
from dataclasses import dataclass
from uuid import UUID

from openai import OpenAI
from sqlalchemy import update
from sqlalchemy.orm import Session

from src.models import Chunk

logger = logging.getLogger(__name__)


@dataclass
class ReconstructedSection:
    """A reconstructed section from linked chunks."""

    text: str
    chunk_ids: list[UUID]
    source_chunk_id: UUID  # The chunk that triggered this reconstruction
    relevance_score: float


BOUNDARY_DETECTION_PROMPT = """Examine this legal text and determine if it forms a complete section.

A complete section:
- Begins at a natural boundary (section number, heading, or document start)
- Ends at a natural boundary (before the next section, or at document end)
- Contains a complete legal provision, rule, or coherent unit of information

Text to examine:
---
{text}
---

Answer these two questions:
1. Does this text BEGIN at a natural section boundary? (YES or NO)
2. Does this text END at a natural section boundary? (YES or NO)

Respond in exactly this format:
START_BOUNDARY: YES or NO
END_BOUNDARY: YES or NO"""


def check_section_boundaries(
    text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> tuple[bool, bool]:
    """
    Use LLM to determine if text has complete section boundaries.

    Args:
        text: The text to check
        api_key: OpenAI API key
        model: Model to use for boundary detection

    Returns:
        Tuple of (starts_at_boundary, ends_at_boundary)
    """
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": BOUNDARY_DETECTION_PROMPT.format(text=text[:4000])}
            ],
            temperature=0.0,
            max_completion_tokens=50,
        )

        content = response.choices[0].message.content.strip().upper()

        starts_at_boundary = "START_BOUNDARY: YES" in content
        ends_at_boundary = "END_BOUNDARY: YES" in content

        return starts_at_boundary, ends_at_boundary

    except Exception as e:
        logger.warning(f"Boundary detection failed: {e}")
        # Default to assuming boundaries are incomplete
        return False, False


def fetch_chunk_by_id(session: Session, chunk_id: UUID) -> Chunk | None:
    """Fetch a chunk by its ID."""
    return session.query(Chunk).filter(Chunk.id == chunk_id).first()


def cache_boundary_detection(
    session: Session,
    chunk_id: UUID,
    is_section_start: bool,
    is_section_end: bool,
) -> None:
    """Cache boundary detection results for a chunk."""
    session.execute(
        update(Chunk)
        .where(Chunk.id == chunk_id)
        .values(
            is_section_start=1 if is_section_start else 0,
            is_section_end=1 if is_section_end else 0,
        )
    )


def reconstruct_section(
    session: Session,
    initial_chunk: Chunk,
    api_key: str,
    max_extension: int = 5,
    boundary_model: str = "gpt-4o-mini",
) -> ReconstructedSection:
    """
    Reconstruct a complete section by extending from an initial chunk.

    Args:
        session: Database session
        initial_chunk: The chunk to start reconstruction from
        api_key: OpenAI API key
        max_extension: Maximum chunks to extend in each direction
        boundary_model: Model to use for boundary detection

    Returns:
        ReconstructedSection containing the complete section text
    """
    # Check if we already have cached boundary info
    if initial_chunk.is_section_start == 1 and initial_chunk.is_section_end == 1:
        # This chunk is already a complete section
        return ReconstructedSection(
            text=initial_chunk.chunk_text,
            chunk_ids=[initial_chunk.id],
            source_chunk_id=initial_chunk.id,
            relevance_score=0.0,  # Will be set by caller
        )

    # Collect chunks for the section
    section_chunks = [initial_chunk]
    chunk_ids = [initial_chunk.id]

    # Extend backward
    current = initial_chunk
    for _ in range(max_extension):
        if not current.prev_chunk_id:
            # Reached document start
            break

        prev_chunk = fetch_chunk_by_id(session, current.prev_chunk_id)
        if not prev_chunk:
            break

        # Check if prev_chunk is from the same document
        if prev_chunk.document_id != initial_chunk.document_id:
            break

        section_chunks.insert(0, prev_chunk)
        chunk_ids.insert(0, prev_chunk.id)

        # Check if we've reached a section boundary
        combined_text = "\n".join(c.chunk_text for c in section_chunks)
        starts_at_boundary, _ = check_section_boundaries(
            combined_text, api_key, boundary_model
        )

        if starts_at_boundary:
            # Cache the boundary detection
            cache_boundary_detection(session, prev_chunk.id, True, False)
            break

        current = prev_chunk

    # Extend forward
    current = initial_chunk
    for _ in range(max_extension):
        if not current.next_chunk_id:
            # Reached document end
            break

        next_chunk = fetch_chunk_by_id(session, current.next_chunk_id)
        if not next_chunk:
            break

        # Check if next_chunk is from the same document
        if next_chunk.document_id != initial_chunk.document_id:
            break

        section_chunks.append(next_chunk)
        chunk_ids.append(next_chunk.id)

        # Check if we've reached a section boundary
        combined_text = "\n".join(c.chunk_text for c in section_chunks)
        _, ends_at_boundary = check_section_boundaries(
            combined_text, api_key, boundary_model
        )

        if ends_at_boundary:
            # Cache the boundary detection
            cache_boundary_detection(session, next_chunk.id, False, True)
            break

        current = next_chunk

    # Combine all chunks into section text
    section_text = "\n".join(c.chunk_text for c in section_chunks)

    return ReconstructedSection(
        text=section_text,
        chunk_ids=chunk_ids,
        source_chunk_id=initial_chunk.id,
        relevance_score=0.0,  # Will be set by caller
    )


def reconstruct_sections_for_retrieval(
    session: Session,
    chunks: list[Chunk],
    scores: list[float],
    api_key: str,
    max_extension: int = 5,
    boundary_model: str = "gpt-4o-mini",
) -> list[ReconstructedSection]:
    """
    Reconstruct complete sections for a list of retrieved chunks.

    Handles deduplication when multiple chunks are from the same section.

    Args:
        session: Database session
        chunks: Retrieved chunks from vector search
        scores: Relevance scores for each chunk
        api_key: OpenAI API key
        max_extension: Maximum chunks to extend in each direction
        boundary_model: Model to use for boundary detection

    Returns:
        List of reconstructed sections, deduplicated
    """
    sections = []
    seen_chunk_ids = set()

    for chunk, score in zip(chunks, scores):
        if chunk.id in seen_chunk_ids:
            continue

        section = reconstruct_section(
            session=session,
            initial_chunk=chunk,
            api_key=api_key,
            max_extension=max_extension,
            boundary_model=boundary_model,
        )
        section.relevance_score = score

        # Mark all chunks in this section as seen
        seen_chunk_ids.update(section.chunk_ids)

        sections.append(section)

    # Commit cached boundary detections
    try:
        session.commit()
    except Exception as e:
        logger.warning(f"Failed to cache boundary detections: {e}")
        session.rollback()

    return sections
