"""Integration tests for database connections."""

import pytest
from qdrant_client.models import Distance, VectorParams
from sqlalchemy import text

from src.config import Settings
from src.database import (
    check_postgres_health,
    check_qdrant_health,
    create_qdrant_collection,
    get_postgres_engine,
    get_postgres_session,
    get_qdrant_client,
    init_postgres_schema,
)
from src.models import ChunkingStrategy


class TestQdrantConnection:
    """Tests for Qdrant client connection."""

    def test_qdrant_client_connects_in_memory(self):
        """Qdrant client should connect in memory mode for testing."""
        client = get_qdrant_client(in_memory=True)

        # Should be able to list collections
        collections = client.get_collections()
        assert collections is not None

    def test_create_collection_for_chunking_strategy(self):
        """Should create a collection for a chunking strategy."""
        client = get_qdrant_client(in_memory=True)

        create_qdrant_collection(
            client=client,
            strategy=ChunkingStrategy.RECURSIVE,
            vector_size=1536,
        )

        # Collection should exist
        assert client.collection_exists("legal_chunks_recursive")

        # Collection should have correct config
        info = client.get_collection("legal_chunks_recursive")
        assert info.config.params.vectors.size == 1536
        assert info.config.params.vectors.distance == Distance.COSINE

    def test_create_all_strategy_collections(self):
        """Should create collections for all chunking strategies."""
        client = get_qdrant_client(in_memory=True)

        for strategy in ChunkingStrategy:
            create_qdrant_collection(
                client=client,
                strategy=strategy,
                vector_size=1536,
            )

        # All collections should exist
        assert client.collection_exists("legal_chunks_fixed")
        assert client.collection_exists("legal_chunks_paragraph")
        assert client.collection_exists("legal_chunks_recursive")

    def test_create_collection_idempotent(self):
        """Creating a collection twice should not raise an error."""
        client = get_qdrant_client(in_memory=True)

        create_qdrant_collection(client, ChunkingStrategy.FIXED, 1536)
        create_qdrant_collection(client, ChunkingStrategy.FIXED, 1536)

        assert client.collection_exists("legal_chunks_fixed")

    def test_check_qdrant_health_in_memory(self):
        """Health check should pass for in-memory client."""
        client = get_qdrant_client(in_memory=True)

        is_healthy = check_qdrant_health(client)
        assert is_healthy is True


class TestPostgresConnection:
    """Tests for PostgreSQL connection."""

    @pytest.fixture
    def sqlite_settings(self, monkeypatch):
        """Use SQLite for testing instead of PostgreSQL."""
        monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
        return Settings()

    def test_postgres_engine_creates(self):
        """Engine should be created from settings."""
        # Use SQLite for testing
        engine = get_postgres_engine(url="sqlite:///:memory:")

        assert engine is not None

        # Should be able to connect
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_postgres_session_works(self):
        """Session should allow database operations."""
        engine = get_postgres_engine(url="sqlite:///:memory:")

        with get_postgres_session(engine) as session:
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_init_schema_creates_tables(self):
        """Schema initialization should create all tables."""
        engine = get_postgres_engine(url="sqlite:///:memory:")

        init_postgres_schema(engine)

        # Check tables exist
        with engine.connect() as conn:
            # SQLite uses sqlite_master instead of information_schema
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = {row[0] for row in result}

        expected_tables = {
            "documents",
            "chunks",
            "query_logs",
            "evaluation_runs",
            "ingestion_jobs",
            "test_queries",
        }
        assert expected_tables.issubset(tables)

    def test_check_postgres_health(self):
        """Health check should pass for valid connection."""
        engine = get_postgres_engine(url="sqlite:///:memory:")

        is_healthy = check_postgres_health(engine)
        assert is_healthy is True

    def test_check_postgres_health_bad_connection(self):
        """Health check should fail for invalid connection."""
        engine = get_postgres_engine(url="sqlite:///nonexistent/path/db.sqlite")

        is_healthy = check_postgres_health(engine)
        assert is_healthy is False
