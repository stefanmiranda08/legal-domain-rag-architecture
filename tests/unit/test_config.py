"""Tests for configuration management."""

import pytest

from src.config import Settings


class TestSettings:
    """Tests for Settings class."""

    def test_settings_loads_from_env(self):
        """Settings should load values from environment variables."""
        settings = Settings()

        assert settings.openai_api_key == "test-key-not-real"
        assert settings.postgres_host == "localhost"
        assert settings.postgres_port == 5432
        assert settings.postgres_user == "testuser"
        assert settings.postgres_password == "testpass"
        assert settings.postgres_db == "testdb"
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333

    def test_settings_has_default_embedding_model(self):
        """Settings should have default embedding model."""
        settings = Settings()

        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.embedding_dimensions == 1536

    def test_settings_has_default_llm_model(self):
        """Settings should have default LLM model."""
        settings = Settings()

        assert settings.llm_model == "gpt-4o-mini"

    def test_settings_has_chunking_defaults(self):
        """Settings should have default chunking parameters."""
        settings = Settings()

        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 50

    def test_postgres_url_construction(self):
        """Settings should construct valid PostgreSQL URL."""
        settings = Settings()

        expected = "postgresql://testuser:testpass@localhost:5432/testdb"
        assert settings.postgres_url == expected

    def test_qdrant_url_construction(self):
        """Settings should construct valid Qdrant URL."""
        settings = Settings()

        expected = "http://localhost:6333"
        assert settings.qdrant_url == expected
