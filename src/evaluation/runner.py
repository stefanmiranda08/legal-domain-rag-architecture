"""Evaluation runner for comparing chunking strategies."""

import time
from dataclasses import asdict
from datetime import datetime
from uuid import uuid4

from qdrant_client import QdrantClient
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config import Settings, get_settings
from src.database import get_postgres_session
from src.evaluation.metrics import (
    AggregatedMetrics,
    EvaluationResult,
    calculate_metrics,
    mean_reciprocal_rank,
    recall_at_k,
)
from src.models import ChunkingStrategy, EvaluationRun, PerQueryResult
from src.retrieval import search_chunks


def get_test_queries(
    engine: Engine,
    test_set_id: str = "default",
) -> list[dict]:
    """
    Retrieve test queries from the database.

    Args:
        engine: PostgreSQL engine.
        test_set_id: ID of the test set to retrieve.

    Returns:
        List of test query dicts with 'id', 'query_text', 'expected_doc_ids'.
    """
    with get_postgres_session(engine) as session:
        result = session.execute(
            text(
                "SELECT id, query_text, expected_document_ids "
                "FROM test_queries WHERE test_set_id = :test_set_id"
            ),
            {"test_set_id": test_set_id},
        )

        return [
            {
                "id": row[0],
                "query_text": row[1],
                "expected_doc_ids": row[2] or [],
            }
            for row in result
        ]


def run_evaluation(
    qdrant_client: QdrantClient,
    postgres_engine: Engine,
    strategy: ChunkingStrategy,
    test_set_id: str = "default",
    openai_api_key: str | None = None,
    settings: Settings | None = None,
    top_k: int = 10,
) -> dict:
    """
    Run evaluation on a chunking strategy.

    Args:
        qdrant_client: Qdrant client.
        postgres_engine: PostgreSQL engine.
        strategy: Chunking strategy to evaluate.
        test_set_id: ID of the test set.
        openai_api_key: OpenAI API key.
        settings: Optional settings.
        top_k: Number of results to retrieve.

    Returns:
        Evaluation results dict.
    """
    settings = settings or get_settings()
    openai_api_key = openai_api_key or settings.openai_api_key
    evaluation_id = f"eval_{uuid4().hex[:8]}"

    # Get test queries
    test_queries = get_test_queries(postgres_engine, test_set_id)

    if not test_queries:
        # Return empty results if no test queries
        return {
            "evaluation_id": evaluation_id,
            "strategy": strategy.value,
            "metrics": {
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "mrr": 0.0,
                "avg_latency_ms": 0.0,
            },
            "per_query_results": [],
        }

    # Run retrieval for each query
    query_results = []
    per_query_results = []

    for test_query in test_queries:
        start_time = time.time()

        # Search for chunks
        chunks = search_chunks(
            query=test_query["query_text"],
            qdrant_client=qdrant_client,
            strategy=strategy,
            openai_api_key=openai_api_key,
            top_k=top_k,
            settings=settings,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Extract retrieved document IDs
        retrieved_doc_ids = [chunk.document_id for chunk in chunks]
        expected_doc_ids = test_query["expected_doc_ids"]

        # Calculate per-query metrics
        r5 = recall_at_k(retrieved_doc_ids, expected_doc_ids, k=5)
        r10 = recall_at_k(retrieved_doc_ids, expected_doc_ids, k=10)
        mrr = mean_reciprocal_rank(retrieved_doc_ids, expected_doc_ids)

        query_results.append(
            {
                "retrieved": retrieved_doc_ids,
                "expected": expected_doc_ids,
                "latency_ms": latency_ms,
            }
        )

        per_query_results.append(
            PerQueryResult(
                query_id=test_query["id"],
                query=test_query["query_text"],
                expected_doc_ids=expected_doc_ids,
                retrieved_doc_ids=retrieved_doc_ids[:5],  # Top 5 for display
                recall_at_5=r5,
            )
        )

    # Calculate aggregated metrics
    metrics = calculate_metrics(query_results)

    # Store evaluation results
    _store_evaluation_results(
        postgres_engine,
        evaluation_id=evaluation_id,
        strategy=strategy,
        test_set_id=test_set_id,
        metrics=metrics,
        per_query_results=per_query_results,
    )

    return {
        "evaluation_id": evaluation_id,
        "strategy": strategy.value,
        "metrics": dict(metrics),
        "per_query_results": [r.model_dump() for r in per_query_results],
    }


def _store_evaluation_results(
    engine: Engine,
    evaluation_id: str,
    strategy: ChunkingStrategy,
    test_set_id: str,
    metrics: AggregatedMetrics,
    per_query_results: list[PerQueryResult],
) -> None:
    """Store evaluation results in the database."""
    try:
        with get_postgres_session(engine) as session:
            run = EvaluationRun(
                id=uuid4(),
                chunking_strategy=strategy.value,
                test_set_id=test_set_id,
                recall_at_5=metrics["recall_at_5"],
                recall_at_10=metrics["recall_at_10"],
                mrr=metrics["mrr"],
                avg_latency_ms=metrics["avg_latency_ms"],
                per_query_results=[r.model_dump() for r in per_query_results],
            )
            session.add(run)
    except Exception:
        # Don't fail if storage fails
        pass


def compare_strategies(
    qdrant_client: QdrantClient,
    postgres_engine: Engine,
    strategies: list[ChunkingStrategy],
    test_set_id: str = "default",
    openai_api_key: str | None = None,
    settings: Settings | None = None,
) -> dict[str, dict]:
    """
    Run evaluation on multiple strategies for comparison.

    Args:
        qdrant_client: Qdrant client.
        postgres_engine: PostgreSQL engine.
        strategies: List of strategies to evaluate.
        test_set_id: ID of the test set.
        openai_api_key: OpenAI API key.
        settings: Optional settings.

    Returns:
        Dict mapping strategy name to evaluation results.
    """
    results = {}

    for strategy in strategies:
        results[strategy.value] = run_evaluation(
            qdrant_client=qdrant_client,
            postgres_engine=postgres_engine,
            strategy=strategy,
            test_set_id=test_set_id,
            openai_api_key=openai_api_key,
            settings=settings,
        )

    return results
