"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    calculate_metrics,
    EvaluationResult,
)


class TestRecallAtK:
    """Tests for recall@k metric."""

    def test_perfect_recall(self):
        """Should return 1.0 when all expected docs are retrieved."""
        retrieved = ["doc1", "doc2", "doc3"]
        expected = ["doc1", "doc2"]

        result = recall_at_k(retrieved, expected, k=3)

        assert result == 1.0

    def test_partial_recall(self):
        """Should return fraction when some expected docs are retrieved."""
        retrieved = ["doc1", "doc3", "doc4"]
        expected = ["doc1", "doc2"]

        result = recall_at_k(retrieved, expected, k=3)

        assert result == 0.5

    def test_zero_recall(self):
        """Should return 0.0 when no expected docs are retrieved."""
        retrieved = ["doc3", "doc4", "doc5"]
        expected = ["doc1", "doc2"]

        result = recall_at_k(retrieved, expected, k=3)

        assert result == 0.0

    def test_respects_k_limit(self):
        """Should only consider first k retrieved docs."""
        retrieved = ["doc3", "doc4", "doc1", "doc2"]  # doc1, doc2 after position k
        expected = ["doc1", "doc2"]

        result = recall_at_k(retrieved, expected, k=2)

        assert result == 0.0

    def test_empty_expected(self):
        """Should return 0.0 for empty expected list."""
        retrieved = ["doc1", "doc2"]
        expected = []

        result = recall_at_k(retrieved, expected, k=5)

        assert result == 0.0

    def test_empty_retrieved(self):
        """Should return 0.0 for empty retrieved list."""
        retrieved = []
        expected = ["doc1", "doc2"]

        result = recall_at_k(retrieved, expected, k=5)

        assert result == 0.0


class TestMeanReciprocalRank:
    """Tests for MRR metric."""

    def test_first_position(self):
        """Should return 1.0 when first relevant doc is at position 1."""
        retrieved = ["doc1", "doc2", "doc3"]
        expected = ["doc1"]

        result = mean_reciprocal_rank(retrieved, expected)

        assert result == 1.0

    def test_second_position(self):
        """Should return 0.5 when first relevant doc is at position 2."""
        retrieved = ["doc2", "doc1", "doc3"]
        expected = ["doc1"]

        result = mean_reciprocal_rank(retrieved, expected)

        assert result == 0.5

    def test_third_position(self):
        """Should return 0.33 when first relevant doc is at position 3."""
        retrieved = ["doc2", "doc3", "doc1"]
        expected = ["doc1"]

        result = mean_reciprocal_rank(retrieved, expected)

        assert abs(result - 1 / 3) < 0.01

    def test_no_relevant_docs(self):
        """Should return 0.0 when no relevant docs are found."""
        retrieved = ["doc2", "doc3", "doc4"]
        expected = ["doc1"]

        result = mean_reciprocal_rank(retrieved, expected)

        assert result == 0.0

    def test_empty_lists(self):
        """Should return 0.0 for empty lists."""
        assert mean_reciprocal_rank([], ["doc1"]) == 0.0
        assert mean_reciprocal_rank(["doc1"], []) == 0.0


class TestCalculateMetrics:
    """Tests for aggregated metrics calculation."""

    def test_calculates_all_metrics(self):
        """Should calculate recall@5, recall@10, and MRR."""
        query_results = [
            {
                "retrieved": ["doc1", "doc2", "doc3", "doc4", "doc5"],
                "expected": ["doc1", "doc3"],
                "latency_ms": 100,
            },
            {
                "retrieved": ["doc2", "doc1", "doc4", "doc5", "doc6"],
                "expected": ["doc1"],
                "latency_ms": 150,
            },
        ]

        metrics = calculate_metrics(query_results)

        assert "recall_at_5" in metrics
        assert "recall_at_10" in metrics
        assert "mrr" in metrics
        assert "avg_latency_ms" in metrics

    def test_averages_across_queries(self):
        """Should average metrics across all queries."""
        query_results = [
            {
                "retrieved": ["doc1"],
                "expected": ["doc1"],
                "latency_ms": 100,
            },
            {
                "retrieved": ["doc2"],
                "expected": ["doc1"],
                "latency_ms": 200,
            },
        ]

        metrics = calculate_metrics(query_results)

        # First query: recall=1.0, MRR=1.0
        # Second query: recall=0.0, MRR=0.0
        # Average: recall=0.5, MRR=0.5
        assert metrics["recall_at_5"] == 0.5
        assert metrics["mrr"] == 0.5
        assert metrics["avg_latency_ms"] == 150.0

    def test_empty_results(self):
        """Should handle empty results list."""
        metrics = calculate_metrics([])

        assert metrics["recall_at_5"] == 0.0
        assert metrics["recall_at_10"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["avg_latency_ms"] == 0.0


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_result_creation(self):
        """EvaluationResult should store all fields."""
        result = EvaluationResult(
            query_id="q1",
            query_text="test query",
            expected_doc_ids=["doc1"],
            retrieved_doc_ids=["doc1", "doc2"],
            recall_at_5=1.0,
            recall_at_10=1.0,
            mrr=1.0,
            latency_ms=100,
        )

        assert result.query_id == "q1"
        assert result.recall_at_5 == 1.0
