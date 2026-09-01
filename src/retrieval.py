"""Retrieval module for vector search with metadata filtering."""

from dataclasses import dataclass
from datetime import date

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import DatetimeRange, FieldCondition, Filter, MatchValue
from sqlalchemy.engine import Engine

from src.config import Settings, get_settings
from src.database import get_collection_name, get_postgres_session
from src.models import Chunk, ChunkingStrategy, QueryFilters


@dataclass
class RetrievedChunk:
    """A chunk retrieved from vector search."""

    chunk_id: str
    document_id: str
    text: str
    citation: str
    score: float
    chunk_index: int


def build_qdrant_filter(filters: QueryFilters) -> Filter | None:
    """
    Build a Qdrant filter from QueryFilters.

    Args:
        filters: Query filters to apply.

    Returns:
        Qdrant Filter object, or None if no filters specified.
    """
    conditions = []

    if filters.jurisdiction is not None:
        conditions.append(
            FieldCondition(
                key="jurisdiction",
                match=MatchValue(value=filters.jurisdiction.value),
            )
        )

    if filters.document_type is not None:
        conditions.append(
            FieldCondition(
                key="document_type",
                match=MatchValue(value=filters.document_type.value),
            )
        )

    if filters.date_from is not None:
        conditions.append(
            FieldCondition(
                key="date",
                range=DatetimeRange(gte=filters.date_from.isoformat()),
            )
        )

    if filters.date_to is not None:
        conditions.append(
            FieldCondition(
                key="date",
                range=DatetimeRange(lte=filters.date_to.isoformat()),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)


def generate_query_embedding(
    query: str,
    api_key: str,
    model: str = "text-embedding-3-small",
    dimensions: int = 1536,
) -> list[float]:
    """
    Generate embedding for a query string.

    Args:
        query: Query text to embed.
        api_key: OpenAI API key.
        model: Embedding model to use.
        dimensions: Embedding dimensions.

    Returns:
        Embedding vector.
    """
    client = OpenAI(api_key=api_key)

    response = client.embeddings.create(
        input=query,
        model=model,
        dimensions=dimensions,
    )

    return response.data[0].embedding


def get_chunk_texts(
    chunk_ids: list[str],
    engine: Engine | None = None,
) -> dict[str, str]:
    """
    Retrieve chunk texts from PostgreSQL.

    Args:
        chunk_ids: List of Qdrant point IDs.
        engine: SQLAlchemy engine (optional, creates new if not provided).

    Returns:
        Dict mapping chunk_id to chunk text.
    """
    if not chunk_ids or engine is None:
        return {}

    from sqlalchemy import text

    with get_postgres_session(engine) as session:
        # Query chunks by qdrant_point_id
        placeholders = ", ".join([f":id_{i}" for i in range(len(chunk_ids))])
        query = text(
            f"SELECT qdrant_point_id, chunk_text FROM chunks "
            f"WHERE qdrant_point_id::text IN ({placeholders})"
        )

        params = {f"id_{i}": str(cid) for i, cid in enumerate(chunk_ids)}
        result = session.execute(query, params)

        return {str(row[0]): row[1] for row in result}


def search_chunks(
    query: str,
    qdrant_client: QdrantClient,
    strategy: ChunkingStrategy,
    openai_api_key: str,
    filters: QueryFilters | None = None,
    top_k: int = 10,
    postgres_engine: Engine | None = None,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """
    Search for relevant chunks using vector similarity.

    Args:
        query: Natural language query.
        qdrant_client: Qdrant client instance.
        strategy: Chunking strategy to search.
        openai_api_key: OpenAI API key for query embedding.
        filters: Optional metadata filters.
        top_k: Number of results to return.
        postgres_engine: Optional PostgreSQL engine for chunk text retrieval.
        settings: Optional settings override.

    Returns:
        List of retrieved chunks sorted by relevance.
    """
    if not query or not query.strip():
        return []

    settings = settings or get_settings()
    collection_name = get_collection_name(strategy)

    # Generate query embedding
    query_embedding = generate_query_embedding(
        query=query,
        api_key=openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )

    # Build filter if provided
    qdrant_filter = None
    if filters is not None:
        qdrant_filter = build_qdrant_filter(filters)

    # Search Qdrant
    result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    if not result.points:
        return []

    # Get chunk texts from PostgreSQL
    chunk_ids = [str(point.id) for point in result.points]
    chunk_texts = get_chunk_texts(chunk_ids, postgres_engine)

    # Build retrieved chunks
    chunks = []
    for point in result.points:
        payload = point.payload or {}
        chunk_id = str(point.id)

        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=payload.get("document_id", ""),
                text=chunk_texts.get(chunk_id, ""),
                citation=payload.get("citation", ""),
                score=point.score,
                chunk_index=payload.get("chunk_index", 0),
            )
        )

    return chunks
