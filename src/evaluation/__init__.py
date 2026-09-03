"""Evaluation framework components."""

from src.evaluation.harness import (
    ExperimentConfig,
    EvaluationResult,
    QueryResult,
    run_evaluation as run_ragas_evaluation,
    load_test_queries,
    compare_experiments,
)
from src.evaluation.metrics import (
    recall_at_k,
    mean_reciprocal_rank,
    calculate_metrics,
)
from src.evaluation.runner import (
    run_evaluation,
    compare_strategies,
)

__all__ = [
    # RAGAS-based evaluation (new)
    "ExperimentConfig",
    "EvaluationResult",
    "QueryResult",
    "run_ragas_evaluation",
    "load_test_queries",
    "compare_experiments",
    # Retrieval metrics
    "recall_at_k",
    "mean_reciprocal_rank",
    "calculate_metrics",
    # Strategy comparison
    "run_evaluation",
    "compare_strategies",
]
