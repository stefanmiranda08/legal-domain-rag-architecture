"""
Evaluation harness for RAG system using RAGAS metrics.

This module provides functionality to evaluate the RAG pipeline across
different experimental configurations and compute quality metrics.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import EvaluationDataset, SingleTurnSample

from src.config import Settings
from src.retrieval import search_chunks, RetrievedChunk
from src.generation import generate_answer

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
TEST_QUERIES_PATH = EVALUATION_DIR / "test_queries.json"
RESULTS_DIR = EVALUATION_DIR / "results"


@dataclass
class ExperimentConfig:
    """Configuration for an evaluation experiment."""

    name: str
    chunking_strategy: str = "recursive"
    top_k: int = 10
    llm_model: str = "gpt-4o-mini"
    system_prompt: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryResult:
    """Result from running a single query through the RAG pipeline."""

    query_id: str
    query: str
    category: str
    answer: str
    retrieved_contexts: list[str]
    citations: list[dict]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    tokens_used: int


@dataclass
class EvaluationResult:
    """Complete evaluation result for an experiment."""

    experiment_name: str
    config: dict
    timestamp: str
    metrics: dict = field(default_factory=dict)
    per_query_scores: list[dict] = field(default_factory=list)
    query_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path | None = None) -> Path:
        """Save results to JSON file."""
        if path is None:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{self.experiment_name}_{self.timestamp}.json"
            path = RESULTS_DIR / filename

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return path


def load_test_queries() -> list[dict]:
    """Load test queries from JSON file."""
    with open(TEST_QUERIES_PATH) as f:
        data = json.load(f)
    return data["queries"]


def run_rag_pipeline(
    query: str,
    config: ExperimentConfig,
    settings: Settings,
    qdrant_client,
    postgres_engine,
) -> QueryResult:
    """
    Run a single query through the RAG pipeline.

    Returns the answer, retrieved contexts, and timing information.
    """
    import time

    # Retrieval
    retrieval_start = time.perf_counter()
    chunks = search_chunks(
        query=query,
        qdrant_client=qdrant_client,
        postgres_engine=postgres_engine,
        strategy=config.chunking_strategy,
        top_k=config.top_k,
        openai_api_key=settings.openai_api_key,
        settings=settings,
    )
    retrieval_end = time.perf_counter()
    retrieval_latency = (retrieval_end - retrieval_start) * 1000

    # Generation
    generation_start = time.perf_counter()
    answer = generate_answer(
        query=query,
        chunks=chunks,
        openai_api_key=settings.openai_api_key,
        model=config.llm_model,
        settings=settings,
    )
    generation_end = time.perf_counter()
    generation_latency = (generation_end - generation_start) * 1000

    # Extract contexts from chunks
    contexts = [chunk.text for chunk in chunks]
    citations = [
        {
            "citation": c.citation,
            "document_id": c.document_id,
            "score": c.score,
        }
        for c in answer.citations
    ]

    return QueryResult(
        query_id="",  # Will be set by caller
        query=query,
        category="",  # Will be set by caller
        answer=answer.text,
        retrieved_contexts=contexts,
        citations=citations,
        retrieval_latency_ms=retrieval_latency,
        generation_latency_ms=generation_latency,
        total_latency_ms=retrieval_latency + generation_latency,
        tokens_used=answer.tokens_used,
    )


def compute_ragas_metrics(
    queries: list[str],
    answers: list[str],
    contexts: list[list[str]],
) -> tuple[dict, list[dict]]:
    """
    Compute RAGAS metrics for a set of query-answer-context triples.

    Returns:
        Tuple of (aggregate_metrics, per_query_scores)
    """
    # Build evaluation samples
    samples = []
    for query, answer, context in zip(queries, answers, contexts):
        samples.append(
            SingleTurnSample(
                user_input=query,
                response=answer,
                retrieved_contexts=context,
            )
        )

    dataset = EvaluationDataset(samples=samples)

    # Run evaluation
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    # Extract aggregate metrics
    aggregate = {}
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision"]:
        if metric_name in result:
            aggregate[metric_name] = float(result[metric_name])

    # Extract per-query scores
    per_query = []
    if hasattr(result, "scores") and result.scores:
        for i, scores in enumerate(result.scores):
            per_query.append({
                "query_index": i,
                **{k: float(v) if v is not None else None for k, v in scores.items()}
            })

    return aggregate, per_query


def run_evaluation(
    config: ExperimentConfig,
    settings: Settings,
    qdrant_client,
    postgres_engine,
    queries: list[dict] | None = None,
) -> EvaluationResult:
    """
    Run a complete evaluation experiment.

    Args:
        config: Experiment configuration
        settings: Application settings
        qdrant_client: Qdrant client instance
        postgres_engine: PostgreSQL engine instance
        queries: Optional list of queries (loads from file if not provided)

    Returns:
        EvaluationResult with metrics and per-query details
    """
    if queries is None:
        queries = load_test_queries()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Starting evaluation: {config.name}")
    logger.info(f"Config: {config.to_dict()}")
    logger.info(f"Running {len(queries)} queries...")

    # Run queries through RAG pipeline
    query_results = []
    all_queries = []
    all_answers = []
    all_contexts = []

    for i, q in enumerate(queries):
        logger.info(f"  Query {i+1}/{len(queries)}: {q['query'][:50]}...")

        try:
            result = run_rag_pipeline(
                query=q["query"],
                config=config,
                settings=settings,
                qdrant_client=qdrant_client,
                postgres_engine=postgres_engine,
            )
            result.query_id = q["id"]
            result.category = q["category"]

            query_results.append(asdict(result))
            all_queries.append(q["query"])
            all_answers.append(result.answer)
            all_contexts.append(result.retrieved_contexts)

        except Exception as e:
            logger.error(f"  Error on query {q['id']}: {e}")
            query_results.append({
                "query_id": q["id"],
                "query": q["query"],
                "category": q["category"],
                "error": str(e),
            })

    # Compute RAGAS metrics
    logger.info("Computing RAGAS metrics...")
    aggregate_metrics, per_query_scores = compute_ragas_metrics(
        queries=all_queries,
        answers=all_answers,
        contexts=all_contexts,
    )

    # Add latency metrics
    successful_results = [r for r in query_results if "error" not in r]
    if successful_results:
        latencies = [r["total_latency_ms"] for r in successful_results]
        aggregate_metrics["latency_p50_ms"] = sorted(latencies)[len(latencies) // 2]
        aggregate_metrics["latency_mean_ms"] = sum(latencies) / len(latencies)
        aggregate_metrics["total_tokens"] = sum(r["tokens_used"] for r in successful_results)
        aggregate_metrics["queries_successful"] = len(successful_results)
        aggregate_metrics["queries_failed"] = len(queries) - len(successful_results)

    logger.info(f"Evaluation complete. Metrics: {aggregate_metrics}")

    return EvaluationResult(
        experiment_name=config.name,
        config=config.to_dict(),
        timestamp=timestamp,
        metrics=aggregate_metrics,
        per_query_scores=per_query_scores,
        query_results=query_results,
    )


def compare_experiments(result_files: list[Path]) -> dict:
    """
    Load multiple experiment results and create a comparison summary.

    Args:
        result_files: List of paths to result JSON files

    Returns:
        Comparison dictionary with metrics across experiments
    """
    experiments = []

    for path in result_files:
        with open(path) as f:
            data = json.load(f)
            experiments.append({
                "name": data["experiment_name"],
                "config": data["config"],
                "metrics": data["metrics"],
            })

    return {
        "experiments": experiments,
        "comparison_timestamp": datetime.utcnow().isoformat(),
    }
