"""Evaluation metrics for retrieval quality assessment."""

from dataclasses import dataclass
from typing import TypedDict


@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""

    query_id: str
    query_text: str
    expected_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    recall_at_5: float
    recall_at_10: float
    mrr: float
    latency_ms: int


class AggregatedMetrics(TypedDict):
    """Aggregated metrics across all queries."""

    recall_at_5: float
    recall_at_10: float
    mrr: float
    avg_latency_ms: float


def recall_at_k(
    retrieved: list[str],
    expected: list[str],
    k: int,
) -> float:
    """
    Calculate recall@k.

    Recall@k measures the proportion of relevant documents that appear
    in the top-k retrieved results.

    Args:
        retrieved: List of retrieved document IDs in ranked order.
        expected: List of relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not expected:
        return 0.0

    top_k = set(retrieved[:k])
    expected_set = set(expected)

    relevant_in_top_k = len(top_k & expected_set)

    return relevant_in_top_k / len(expected_set)


def mean_reciprocal_rank(
    retrieved: list[str],
    expected: list[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR is the average of reciprocal ranks of the first relevant document
    across queries. For a single query, it's 1/rank of first relevant doc.

    Args:
        retrieved: List of retrieved document IDs in ranked order.
        expected: List of relevant document IDs.

    Returns:
        MRR score between 0.0 and 1.0.
    """
    if not expected or not retrieved:
        return 0.0

    expected_set = set(expected)

    for i, doc_id in enumerate(retrieved):
        if doc_id in expected_set:
            return 1.0 / (i + 1)

    return 0.0


def calculate_metrics(
    query_results: list[dict],
) -> AggregatedMetrics:
    """
    Calculate aggregated metrics across multiple query results.

    Args:
        query_results: List of dicts with 'retrieved', 'expected', 'latency_ms'.

    Returns:
        Aggregated metrics dictionary.
    """
    if not query_results:
        return AggregatedMetrics(
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
            avg_latency_ms=0.0,
        )

    recall_5_scores = []
    recall_10_scores = []
    mrr_scores = []
    latencies = []

    for result in query_results:
        retrieved = result["retrieved"]
        expected = result["expected"]
        latency = result.get("latency_ms", 0)

        recall_5_scores.append(recall_at_k(retrieved, expected, k=5))
        recall_10_scores.append(recall_at_k(retrieved, expected, k=10))
        mrr_scores.append(mean_reciprocal_rank(retrieved, expected))
        latencies.append(latency)

    return AggregatedMetrics(
        recall_at_5=sum(recall_5_scores) / len(recall_5_scores),
        recall_at_10=sum(recall_10_scores) / len(recall_10_scores),
        mrr=sum(mrr_scores) / len(mrr_scores),
        avg_latency_ms=sum(latencies) / len(latencies),
    )
