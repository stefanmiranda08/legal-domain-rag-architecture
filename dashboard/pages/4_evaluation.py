"""Evaluation results dashboard page."""

import json
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dashboard.styles import apply_global_styles

st.set_page_config(
    page_title="Evaluation Results",
    page_icon="",
    layout="wide",
)

apply_global_styles()

# Paths
RESULTS_DIR = Path(__file__).parent.parent.parent / "evaluation" / "results"

st.title("RAG Evaluation Results")

# =============================================================================
# METRICS EXPLANATION SECTION
# =============================================================================

st.header("Understanding the Metrics")

with st.expander("Click to expand metric descriptions", expanded=True):
    st.markdown("""
### RAGAS Metrics (LLM-as-Judge)

These metrics use a language model to evaluate response quality without requiring ground truth labels.

| Metric | What it Measures | Range | Why it Matters |
|--------|------------------|-------|----------------|
| **Faithfulness** | Whether the answer is factually consistent with the retrieved context | 0-1 | A low score indicates hallucination — the model is making claims not supported by the documents. Critical for legal accuracy. |
| **Answer Relevancy** | Whether the answer addresses the user's question | 0-1 | A low score means the answer is off-topic or doesn't address what was asked, even if factually correct. |
| **Context Precision** | Whether the retrieved documents are relevant to answering the question | 0-1 | A low score means retrieval is bringing back irrelevant documents, wasting context window and potentially confusing the model. |

### Operational Metrics

| Metric | What it Measures | Why it Matters |
|--------|------------------|----------------|
| **Latency (P50)** | Median response time in milliseconds | User experience — how long users wait for answers |
| **Total Tokens** | Total tokens used across all queries | Cost — directly impacts API spend |
| **Success Rate** | Percentage of queries that completed without error | Reliability — system stability |

### Interpreting Results

- **Comparing chunking strategies**: Higher faithfulness + context precision = better retrieval quality
- **Comparing models**: Higher relevancy scores may indicate better instruction following
- **Comparing prompts**: Professional prompt should show higher structure/relevancy scores

**Important**: These are *relative* metrics for comparing configurations. They do not represent absolute accuracy, which would require expert-labeled ground truth.
""")

st.divider()

# =============================================================================
# LOAD RESULTS
# =============================================================================

def load_all_results() -> list[dict]:
    """Load all evaluation result files."""
    if not RESULTS_DIR.exists():
        return []

    results = []
    for filepath in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(filepath) as f:
                data = json.load(f)
                data["_filepath"] = str(filepath)
                results.append(data)
        except Exception as e:
            st.warning(f"Could not load {filepath.name}: {e}")

    return results


results = load_all_results()

if not results:
    st.warning("No evaluation results found.")
    st.info(f"""
To run an evaluation, use:

```bash
python scripts/run_evaluation.py --name my_experiment \\
    --chunking-strategy recursive \\
    --top-k 10 \\
    --llm-model gpt-4o-mini
```

Results will be saved to `evaluation/results/`
""")
    st.stop()

# =============================================================================
# EXPERIMENT SELECTION
# =============================================================================

st.header("Experiment Results")

# Create summary dataframe
summary_data = []
for r in results:
    metrics = r.get("metrics", {})
    config = r.get("config", {})
    summary_data.append({
        "Experiment": r.get("experiment_name", "Unknown"),
        "Timestamp": r.get("timestamp", "Unknown"),
        "Chunking": config.get("chunking_strategy", "N/A"),
        "Top-K": config.get("top_k", "N/A"),
        "Model": config.get("llm_model", "N/A"),
        "Prompt": config.get("system_prompt", "N/A"),
        "Faithfulness": metrics.get("faithfulness", None),
        "Answer Relevancy": metrics.get("answer_relevancy", None),
        "Context Precision": metrics.get("context_precision", None),
        "Latency P50 (ms)": metrics.get("latency_p50_ms", None),
        "Success": metrics.get("queries_successful", 0),
        "Failed": metrics.get("queries_failed", 0),
    })

summary_df = pd.DataFrame(summary_data)

# Display summary table
st.subheader("All Experiments")
st.dataframe(
    summary_df.style.format({
        "Faithfulness": "{:.3f}",
        "Answer Relevancy": "{:.3f}",
        "Context Precision": "{:.3f}",
        "Latency P50 (ms)": "{:.0f}",
    }, na_rep="N/A").background_gradient(
        subset=["Faithfulness", "Answer Relevancy", "Context Precision"],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    ),
    use_container_width=True,
)

st.divider()

# =============================================================================
# COMPARISON CHARTS
# =============================================================================

st.header("Metric Comparison")

if len(results) >= 2:
    # Filter for comparison
    col1, col2 = st.columns(2)

    with col1:
        compare_by = st.selectbox(
            "Compare by",
            ["Chunking Strategy", "Top-K", "Model", "Prompt"],
        )

    with col2:
        metric_to_plot = st.selectbox(
            "Metric",
            ["Faithfulness", "Answer Relevancy", "Context Precision", "Latency P50 (ms)"],
        )

    # Map display names to config keys
    compare_key_map = {
        "Chunking Strategy": "Chunking",
        "Top-K": "Top-K",
        "Model": "Model",
        "Prompt": "Prompt",
    }
    compare_key = compare_key_map[compare_by]

    # Create comparison chart
    chart_df = summary_df[[compare_key, metric_to_plot]].dropna()

    if not chart_df.empty:
        # Group by the comparison variable and take mean if multiple experiments
        chart_df = chart_df.groupby(compare_key)[metric_to_plot].mean().reset_index()
        chart_df = chart_df.set_index(compare_key)

        st.bar_chart(chart_df)
    else:
        st.info("Not enough data to create comparison chart.")
else:
    st.info("Run at least 2 experiments to see comparison charts.")

st.divider()

# =============================================================================
# DETAILED EXPERIMENT VIEW
# =============================================================================

st.header("Detailed Results")

experiment_names = [r.get("experiment_name", "Unknown") for r in results]
selected_experiment = st.selectbox("Select experiment", experiment_names)

selected_result = next((r for r in results if r.get("experiment_name") == selected_experiment), None)

if selected_result:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        config = selected_result.get("config", {})
        st.json(config)

        st.subheader("Aggregate Metrics")
        metrics = selected_result.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                st.metric(metric_name.replace("_", " ").title(), f"{value:.4f}")
            else:
                st.metric(metric_name.replace("_", " ").title(), value)

    with col2:
        st.subheader("Per-Query Scores")

        per_query = selected_result.get("per_query_scores", [])
        if per_query:
            pq_df = pd.DataFrame(per_query)
            st.dataframe(
                pq_df.style.format({
                    col: "{:.3f}" for col in pq_df.columns if col != "query_index"
                }, na_rep="N/A").background_gradient(
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=1,
                ),
                use_container_width=True,
            )
        else:
            st.info("No per-query scores available.")

        st.subheader("Query Details")

        query_results = selected_result.get("query_results", [])
        if query_results:
            # Filter out errored queries
            successful_queries = [q for q in query_results if "error" not in q]

            if successful_queries:
                query_selector = st.selectbox(
                    "Select query",
                    range(len(successful_queries)),
                    format_func=lambda i: f"{successful_queries[i].get('query_id', i)}: {successful_queries[i].get('query', '')[:50]}...",
                )

                selected_query = successful_queries[query_selector]

                st.markdown(f"**Query:** {selected_query.get('query', 'N/A')}")
                st.markdown(f"**Category:** {selected_query.get('category', 'N/A')}")
                st.markdown(f"**Latency:** {selected_query.get('total_latency_ms', 0):.0f}ms")

                with st.expander("View Answer"):
                    st.markdown(selected_query.get("answer", "No answer"))

                with st.expander("View Retrieved Contexts"):
                    contexts = selected_query.get("retrieved_contexts", [])
                    for i, ctx in enumerate(contexts[:5]):  # Show top 5
                        st.markdown(f"**Context {i+1}:**")
                        st.text(ctx[:500] + "..." if len(ctx) > 500 else ctx)
                        st.divider()
            else:
                st.warning("All queries in this experiment failed.")
        else:
            st.info("No query results available.")

st.divider()

# =============================================================================
# EXPORT OPTIONS
# =============================================================================

st.header("Export")

if results:
    col1, col2 = st.columns(2)

    with col1:
        # Export summary as CSV
        csv_data = summary_df.to_csv(index=False)
        st.download_button(
            label="Download Summary CSV",
            data=csv_data,
            file_name="evaluation_summary.csv",
            mime="text/csv",
        )

    with col2:
        # Export all results as JSON
        all_results_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="Download All Results JSON",
            data=all_results_json,
            file_name="evaluation_results.json",
            mime="application/json",
        )
