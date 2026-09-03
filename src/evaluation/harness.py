"""
Evaluation harness for RAG system using LLM-as-judge metrics.

This module provides functionality to evaluate the RAG pipeline across
different experimental configurations and compute quality metrics using
OpenAI as the judge model.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from src.config import Settings
from src.models import ChunkingStrategy
from src.retrieval import search_chunks
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
    strategy_enum = ChunkingStrategy(config.chunking_strategy)
    chunks = search_chunks(
        query=query,
        qdrant_client=qdrant_client,
        postgres_engine=postgres_engine,
        strategy=strategy_enum,
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
        prompt_variant=config.system_prompt,
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


# =============================================================================
# LLM-AS-JUDGE PROMPTS
# =============================================================================

FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of an AI assistant's answer.

Faithfulness measures whether the answer is factually consistent with the provided context documents.
The answer should not contain claims that are not supported by the context.

Context Documents:
{context}

Question: {question}

Answer: {answer}

Evaluate the faithfulness of the answer on a scale of 0.0 to 1.0:
- 1.0: All claims in the answer are fully supported by the context
- 0.5: Some claims are supported, but others are not found in the context
- 0.0: The answer contains significant claims not supported by the context

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reason": "<brief explanation>"}}"""

ANSWER_RELEVANCY_PROMPT = """You are evaluating the relevancy of an AI assistant's answer.

Answer relevancy measures whether the answer actually addresses the user's question.
An irrelevant answer might be factually correct but off-topic.

Question: {question}

Answer: {answer}

Evaluate the relevancy of the answer on a scale of 0.0 to 1.0:
- 1.0: The answer directly and completely addresses the question
- 0.5: The answer partially addresses the question or includes unnecessary information
- 0.0: The answer does not address the question at all

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reason": "<brief explanation>"}}"""

CONTEXT_PRECISION_PROMPT = """You are evaluating the precision of retrieved context documents.

Context precision measures whether the retrieved documents are relevant to answering the question.
High precision means the retrieved documents contain information useful for answering the question.

Question: {question}

Retrieved Context Documents:
{context}

Evaluate the context precision on a scale of 0.0 to 1.0:
- 1.0: All retrieved documents are highly relevant to the question
- 0.5: Some documents are relevant, others are not
- 0.0: The retrieved documents are not relevant to the question

Respond with ONLY a JSON object in this exact format:
{{"score": <float between 0 and 1>, "reason": "<brief explanation>"}}"""


def evaluate_with_llm(
    prompt: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Use LLM to evaluate a metric and return score + reason.
    """
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        return {
            "score": float(result.get("score", 0.0)),
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        logger.warning(f"LLM evaluation failed: {e}")
        return {"score": 0.0, "reason": f"Evaluation error: {str(e)}"}


def compute_llm_metrics(
    queries: list[str],
    answers: list[str],
    contexts: list[list[str]],
    api_key: str,
    judge_model: str = "gpt-4o-mini",
) -> tuple[dict, list[dict]]:
    """
    Compute LLM-as-judge metrics for a set of query-answer-context triples.

    Metrics computed:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are the retrieved documents relevant?

    Returns:
        Tuple of (aggregate_metrics, per_query_scores)
    """
    per_query = []

    for i, (query, answer, context_list) in enumerate(zip(queries, answers, contexts)):
        context_str = "\n\n---\n\n".join(context_list[:5])  # Limit context for judge

        # Evaluate faithfulness
        faithfulness_result = evaluate_with_llm(
            FAITHFULNESS_PROMPT.format(context=context_str, question=query, answer=answer),
            api_key=api_key,
            model=judge_model,
        )

        # Evaluate answer relevancy
        relevancy_result = evaluate_with_llm(
            ANSWER_RELEVANCY_PROMPT.format(question=query, answer=answer),
            api_key=api_key,
            model=judge_model,
        )

        # Evaluate context precision
        precision_result = evaluate_with_llm(
            CONTEXT_PRECISION_PROMPT.format(question=query, context=context_str),
            api_key=api_key,
            model=judge_model,
        )

        per_query.append({
            "query_index": i,
            "faithfulness": faithfulness_result["score"],
            "faithfulness_reason": faithfulness_result["reason"],
            "answer_relevancy": relevancy_result["score"],
            "answer_relevancy_reason": relevancy_result["reason"],
            "context_precision": precision_result["score"],
            "context_precision_reason": precision_result["reason"],
        })

        logger.debug(f"  Query {i+1}: F={faithfulness_result['score']:.2f}, "
                    f"R={relevancy_result['score']:.2f}, P={precision_result['score']:.2f}")

    # Compute aggregates
    if per_query:
        aggregate = {
            "faithfulness": sum(q["faithfulness"] for q in per_query) / len(per_query),
            "answer_relevancy": sum(q["answer_relevancy"] for q in per_query) / len(per_query),
            "context_precision": sum(q["context_precision"] for q in per_query) / len(per_query),
        }
    else:
        aggregate = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
        }

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

    # Compute LLM-as-judge metrics
    logger.info("Computing LLM-as-judge metrics...")
    aggregate_metrics, per_query_scores = compute_llm_metrics(
        queries=all_queries,
        answers=all_answers,
        contexts=all_contexts,
        api_key=settings.openai_api_key,
        judge_model="gpt-4o-mini",  # Use smaller model for judging
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
