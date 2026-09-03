#!/usr/bin/env python3
"""
Run RAG evaluation experiments with RAGAS metrics.

Usage:
    # Run with default config
    python scripts/run_evaluation.py

    # Run with specific config
    python scripts/run_evaluation.py --name "recursive_top10" \
        --chunking-strategy recursive \
        --top-k 10 \
        --llm-model gpt-4o-mini

    # Run multiple experiments
    python scripts/run_evaluation.py --config evaluation/experiments.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.database import get_postgres_engine, get_qdrant_client
from src.evaluation.harness import (
    ExperimentConfig,
    run_evaluation,
    load_test_queries,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation experiments with RAGAS metrics"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="default_experiment",
        help="Name for this experiment",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        choices=["fixed", "paragraph", "recursive"],
        default="recursive",
        help="Chunking strategy to evaluate",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model for generation (default: from settings)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        choices=["default", "professional"],
        default="professional",
        help="System prompt variant to use",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file with multiple experiments",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated in evaluation/results/)",
    )

    return parser.parse_args()


def run_single_experiment(config: ExperimentConfig, settings, limit: int | None = None):
    """Run a single evaluation experiment."""
    logger.info(f"Running experiment: {config.name}")

    # Initialize clients
    qdrant_client = get_qdrant_client(settings=settings)
    postgres_engine = get_postgres_engine(settings.postgres_url)

    # Load queries
    queries = load_test_queries()
    if limit:
        queries = queries[:limit]
        logger.info(f"Limited to {limit} queries")

    # Run evaluation
    result = run_evaluation(
        config=config,
        settings=settings,
        qdrant_client=qdrant_client,
        postgres_engine=postgres_engine,
        queries=queries,
    )

    return result


def run_from_config_file(config_path: str, settings, limit: int | None = None):
    """Run multiple experiments from a config file."""
    with open(config_path) as f:
        config_data = json.load(f)

    results = []
    for exp in config_data.get("experiments", []):
        config = ExperimentConfig(
            name=exp["name"],
            chunking_strategy=exp.get("chunking_strategy", "recursive"),
            top_k=exp.get("top_k", 10),
            llm_model=exp.get("llm_model", settings.llm_model),
            system_prompt=exp.get("system_prompt", "professional"),
        )

        result = run_single_experiment(config, settings, limit)
        results.append(result)

        # Save individual result
        output_path = result.save()
        logger.info(f"Saved result to: {output_path}")

    return results


def main():
    args = parse_args()
    settings = get_settings()

    if args.config:
        # Run from config file
        results = run_from_config_file(args.config, settings, args.limit)
        logger.info(f"Completed {len(results)} experiments")
    else:
        # Run single experiment from CLI args
        config = ExperimentConfig(
            name=args.name,
            chunking_strategy=args.chunking_strategy,
            top_k=args.top_k,
            llm_model=args.llm_model or settings.llm_model,
            system_prompt=args.system_prompt,
        )

        result = run_single_experiment(config, settings, args.limit)

        # Save result
        output_path = Path(args.output) if args.output else None
        saved_path = result.save(output_path)
        logger.info(f"Saved result to: {saved_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Experiment: {result.experiment_name}")
        print(f"Config: {json.dumps(result.config, indent=2)}")
        print("\nMetrics:")
        for metric, value in result.metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        print("=" * 60)


if __name__ == "__main__":
    main()
