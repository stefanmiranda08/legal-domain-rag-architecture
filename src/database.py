"""Database connection management for Qdrant and PostgreSQL."""

from contextlib import contextmanager
from typing import Generator

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings, get_settings
from src.models import Base, ChunkingStrategy


def get_qdrant_client(
    in_memory: bool = False,
    settings: Settings | None = None,
) -> QdrantClient:
    """
    Create a Qdrant client instance.

    Args:
        in_memory: If True, use in-memory storage (for testing).
        settings: Optional settings override.

    Returns:
        Configured QdrantClient instance.
    """
    if in_memory:
        return QdrantClient(":memory:")

    settings = settings or get_settings()
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )


def get_collection_name(strategy: ChunkingStrategy) -> str:
    """Get the Qdrant collection name for a chunking strategy."""
    return f"legal_chunks_{strategy.value}"


def create_qdrant_collection(
    client: QdrantClient,
    strategy: ChunkingStrategy,
    vector_size: int = 1536,
) -> None:
    """
    Create a Qdrant collection for a chunking strategy.

    Creates the collection if it doesn't exist. Idempotent operation.

    Args:
        client: Qdrant client instance.
        strategy: Chunking strategy to create collection for.
        vector_size: Dimension of embedding vectors.
    """
    collection_name = get_collection_name(strategy)

    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def check_qdrant_health(client: QdrantClient) -> bool:
    """
    Check if Qdrant is healthy and reachable.

    Args:
        client: Qdrant client instance.

    Returns:
        True if healthy, False otherwise.
    """
    try:
        client.get_collections()
        return True
    except Exception:
        return False


def get_postgres_engine(
    url: str | None = None,
    settings: Settings | None = None,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> Engine:
    """
    Create a SQLAlchemy engine for PostgreSQL.

    Args:
        url: Optional database URL override.
        settings: Optional settings override.
        pool_size: Connection pool size.
        max_overflow: Max connections beyond pool_size.

    Returns:
        Configured SQLAlchemy Engine.
    """
    if url is None:
        settings = settings or get_settings()
        url = settings.postgres_url

    # SQLite doesn't support pool_size/max_overflow
    if url.startswith("sqlite"):
        return create_engine(url)

    return create_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,  # Verify connections before use
    )


@contextmanager
def get_postgres_session(engine: Engine) -> Generator[Session, None, None]:
    """
    Create a database session context manager.

    Args:
        engine: SQLAlchemy engine.

    Yields:
        Database session that auto-commits on success, rolls back on error.
    """
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_postgres_schema(engine: Engine) -> None:
    """
    Initialize PostgreSQL schema by creating all tables.

    Args:
        engine: SQLAlchemy engine.
    """
    Base.metadata.create_all(engine)


def check_postgres_health(engine: Engine) -> bool:
    """
    Check if PostgreSQL is healthy and reachable.

    Args:
        engine: SQLAlchemy engine.

    Returns:
        True if healthy, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
