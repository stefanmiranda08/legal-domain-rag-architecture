"""Shared test fixtures."""

import os

import pytest


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set required environment variables for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("QDRANT_HOST", "localhost")
    monkeypatch.setenv("QDRANT_PORT", "6333")
