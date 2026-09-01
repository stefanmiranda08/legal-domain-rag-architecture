"""Integration tests for the FastAPI application."""

from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import app, get_qdrant, get_postgres, get_settings


@pytest.fixture
def mock_dependencies():
    """Override dependencies with mocks."""
    mock_qdrant = Mock()
    mock_engine = Mock()
    mock_settings = Mock()
    mock_settings.openai_api_key = "test-key"
    mock_settings.embedding_model = "text-embedding-3-small"
    mock_settings.embedding_dimensions = 1536
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.chunk_size = 512
    mock_settings.chunk_overlap = 50

    app.dependency_overrides[get_qdrant] = lambda: mock_qdrant
    app.dependency_overrides[get_postgres] = lambda: mock_engine
    app.dependency_overrides[get_settings] = lambda: mock_settings

    yield {
        "qdrant": mock_qdrant,
        "engine": mock_engine,
        "settings": mock_settings,
    }

    app.dependency_overrides.clear()


@pytest.fixture
def client(mock_dependencies):
    """Create test client with mocked dependencies."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client, mock_dependencies):
        """Health check should return healthy status."""
        # Mock healthy connections
        mock_dependencies["qdrant"].get_collections.return_value = []

        with patch("src.api.check_postgres_health", return_value=True):
            with patch("src.api.check_qdrant_health", return_value=True):
                response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["qdrant"] == "connected"
        assert data["postgres"] == "connected"

    def test_health_shows_unhealthy_postgres(self, client, mock_dependencies):
        """Health check should show postgres disconnected."""
        with patch("src.api.check_postgres_health", return_value=False):
            with patch("src.api.check_qdrant_health", return_value=True):
                response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["postgres"] == "disconnected"


class TestQueryEndpoint:
    """Tests for /query endpoint."""

    def test_query_returns_answer(self, client, mock_dependencies):
        """Query should return generated answer with citations."""
        from src.models import Citation
        from src.generation import GeneratedAnswer

        with patch("src.api.search_chunks") as mock_search:
            with patch("src.api.generate_answer") as mock_generate:
                # Mock search results
                mock_chunk = Mock()
                mock_chunk.chunk_id = "c1"
                mock_chunk.document_id = "d1"
                mock_chunk.text = "Legal content..."
                mock_chunk.citation = "Case [2020]"
                mock_chunk.score = 0.9
                mock_chunk.chunk_index = 0
                mock_search.return_value = [mock_chunk]

                # Mock generation with real Citation object
                citation = Citation(
                    document_id="d1",
                    citation="Case [2020]",
                    excerpt="Legal content...",
                    relevance_score=0.9,
                )

                mock_answer = GeneratedAnswer(
                    text="The answer is...",
                    citations=[citation],
                    tokens_used=500,
                )
                mock_generate.return_value = mock_answer

                response = client.post(
                    "/query",
                    json={"query": "What is negligence?"},
                )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert data["answer"] == "The answer is..."

    def test_query_with_filters(self, client, mock_dependencies):
        """Query should accept and apply filters."""
        with patch("src.api.search_chunks") as mock_search:
            with patch("src.api.generate_answer") as mock_generate:
                mock_search.return_value = []
                mock_answer = Mock()
                mock_answer.text = "No results"
                mock_answer.citations = []
                mock_answer.tokens_used = 0
                mock_generate.return_value = mock_answer

                response = client.post(
                    "/query",
                    json={
                        "query": "What is negligence?",
                        "filters": {
                            "jurisdiction": "new_south_wales",
                            "document_type": "decision",
                        },
                    },
                )

        assert response.status_code == 200
        # Verify filters were passed to search
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["filters"].jurisdiction.value == "new_south_wales"

    def test_query_validates_empty_query(self, client, mock_dependencies):
        """Query should reject empty query string."""
        response = client.post(
            "/query",
            json={"query": ""},
        )

        assert response.status_code == 422  # Validation error


class TestIngestEndpoint:
    """Tests for /ingest endpoint."""

    def test_ingest_starts_job(self, client, mock_dependencies):
        """Ingest should start a background job."""
        with patch("src.api.BackgroundTasks") as mock_bg:
            response = client.post(
                "/ingest",
                json={
                    "chunking_strategies": ["recursive"],
                    "limit": 100,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "started"


class TestEvaluateEndpoint:
    """Tests for /evaluate endpoint."""

    def test_evaluate_returns_metrics(self, client, mock_dependencies):
        """Evaluate should return evaluation metrics."""
        with patch("src.api.run_evaluation") as mock_eval:
            mock_eval.return_value = {
                "evaluation_id": "eval_123",
                "strategy": "recursive",
                "metrics": {
                    "recall_at_5": 0.75,
                    "recall_at_10": 0.85,
                    "mrr": 0.68,
                    "avg_latency_ms": 150.0,
                },
                "per_query_results": [],
            }

            response = client.post(
                "/evaluate",
                json={
                    "chunking_strategy": "recursive",
                    "test_set_id": "default",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert data["metrics"]["recall_at_5"] == 0.75
