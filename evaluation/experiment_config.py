"""
Experiment configuration definitions.

This module defines the experimental variables and provides
utilities for generating experiment combinations.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any


# =============================================================================
# EXPERIMENTAL VARIABLES
# =============================================================================

# Chunking strategies available in the system
CHUNKING_STRATEGIES = ["fixed", "paragraph", "recursive"]

# Retrieval parameters
TOP_K_VALUES = [5, 10, 15, 20]

# LLM models for generation
LLM_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5.4",
]

# System prompt variants
SYSTEM_PROMPTS = {
    "default": "default",
    "professional": "professional",
}


# =============================================================================
# EXPERIMENT GENERATION
# =============================================================================

@dataclass
class ExperimentVariable:
    """Definition of an experimental variable."""
    name: str
    values: list[Any]
    description: str


# Define all variables
VARIABLES = [
    ExperimentVariable(
        name="chunking_strategy",
        values=CHUNKING_STRATEGIES,
        description="Strategy used to split documents into chunks for retrieval",
    ),
    ExperimentVariable(
        name="top_k",
        values=TOP_K_VALUES,
        description="Number of chunks retrieved for context",
    ),
    ExperimentVariable(
        name="llm_model",
        values=LLM_MODELS,
        description="LLM model used for answer generation",
    ),
    ExperimentVariable(
        name="system_prompt",
        values=list(SYSTEM_PROMPTS.keys()),
        description="System prompt variant controlling response style",
    ),
]


def generate_experiment_grid(
    chunking_strategies: list[str] | None = None,
    top_k_values: list[int] | None = None,
    llm_models: list[str] | None = None,
    system_prompts: list[str] | None = None,
) -> list[dict]:
    """
    Generate a grid of experiment configurations.

    Args:
        chunking_strategies: List of strategies to test (default: all)
        top_k_values: List of top_k values to test (default: all)
        llm_models: List of models to test (default: all)
        system_prompts: List of prompt variants to test (default: all)

    Returns:
        List of experiment configuration dicts
    """
    strategies = chunking_strategies or CHUNKING_STRATEGIES
    top_ks = top_k_values or TOP_K_VALUES
    models = llm_models or LLM_MODELS
    prompts = system_prompts or list(SYSTEM_PROMPTS.keys())

    experiments = []
    for strategy, top_k, model, prompt in product(strategies, top_ks, models, prompts):
        name = f"{strategy}_k{top_k}_{model.replace('-', '_')}_{prompt}"
        experiments.append({
            "name": name,
            "chunking_strategy": strategy,
            "top_k": top_k,
            "llm_model": model,
            "system_prompt": prompt,
        })

    return experiments


def generate_single_variable_experiments(
    variable: str,
    values: list[Any],
    base_config: dict | None = None,
) -> list[dict]:
    """
    Generate experiments varying a single variable while holding others constant.

    Args:
        variable: Name of variable to vary
        values: Values to test for that variable
        base_config: Base configuration for other variables

    Returns:
        List of experiment configuration dicts
    """
    base = base_config or {
        "chunking_strategy": "recursive",
        "top_k": 10,
        "llm_model": "gpt-4o-mini",
        "system_prompt": "professional",
    }

    experiments = []
    for value in values:
        config = base.copy()
        config[variable] = value
        config["name"] = f"{variable}_{value}".replace("-", "_").replace(".", "_")
        experiments.append(config)

    return experiments


# =============================================================================
# PRESET EXPERIMENT SUITES
# =============================================================================

def get_chunking_comparison_suite() -> list[dict]:
    """Experiments comparing chunking strategies with other variables fixed."""
    return generate_single_variable_experiments(
        variable="chunking_strategy",
        values=CHUNKING_STRATEGIES,
    )


def get_top_k_comparison_suite() -> list[dict]:
    """Experiments comparing top_k values with other variables fixed."""
    return generate_single_variable_experiments(
        variable="top_k",
        values=TOP_K_VALUES,
    )


def get_model_comparison_suite() -> list[dict]:
    """Experiments comparing LLM models with other variables fixed."""
    return generate_single_variable_experiments(
        variable="llm_model",
        values=LLM_MODELS,
    )


def get_quick_validation_suite() -> list[dict]:
    """Minimal suite for quick validation (3 experiments)."""
    return [
        {
            "name": "baseline_recursive",
            "chunking_strategy": "recursive",
            "top_k": 10,
            "llm_model": "gpt-4o-mini",
            "system_prompt": "professional",
        },
        {
            "name": "baseline_fixed",
            "chunking_strategy": "fixed",
            "top_k": 10,
            "llm_model": "gpt-4o-mini",
            "system_prompt": "professional",
        },
        {
            "name": "baseline_paragraph",
            "chunking_strategy": "paragraph",
            "top_k": 10,
            "llm_model": "gpt-4o-mini",
            "system_prompt": "professional",
        },
    ]


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

def export_experiments_json(experiments: list[dict], filepath: str) -> None:
    """Export experiments to JSON file."""
    import json
    with open(filepath, "w") as f:
        json.dump({"experiments": experiments}, f, indent=2)


def get_variable_descriptions() -> dict[str, str]:
    """Get descriptions of all experimental variables."""
    return {v.name: v.description for v in VARIABLES}
