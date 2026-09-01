"""Chunking strategy comparison page."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Chunking Comparison",
    page_icon="",
    layout="wide",
)

st.title("Chunking Strategy Comparison")
st.markdown("Compare retrieval performance across different chunking strategies.")


def get_mock_evaluation_data() -> pd.DataFrame:
    """Generate mock evaluation data for demonstration."""
    return pd.DataFrame({
        "Strategy": ["fixed", "paragraph", "recursive"],
        "Recall@5": [0.72, 0.68, 0.78],
        "Recall@10": [0.85, 0.82, 0.91],
        "MRR": [0.65, 0.61, 0.73],
        "Avg Latency (ms)": [145, 132, 158],
    })


def get_mock_per_query_results() -> pd.DataFrame:
    """Generate mock per-query results."""
    queries = [
        "What constitutes negligence under Australian law?",
        "Define reasonable care in tort law",
        "Explain vicarious liability",
        "What is the duty of care?",
        "Define causation in negligence cases",
    ]

    data = []
    for i, query in enumerate(queries):
        for strategy in ["fixed", "paragraph", "recursive"]:
            recall = 0.6 + (0.2 * (hash(query + strategy) % 3) / 2)
            data.append({
                "Query": query[:50] + "..." if len(query) > 50 else query,
                "Strategy": strategy,
                "Recall@5": round(recall, 2),
            })

    return pd.DataFrame(data)


# Sidebar filters
st.sidebar.header("Filters")

test_set = st.sidebar.selectbox(
    "Test Set",
    ["default", "negligence", "contract_law"],
    help="Select the evaluation test set",
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
    help="Filter evaluations by date range",
)

# Main content
st.header("Strategy Performance Overview")

# Get evaluation data
eval_df = get_mock_evaluation_data()

# Display metrics comparison
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Recall@5")
    st.bar_chart(eval_df.set_index("Strategy")["Recall@5"])

with col2:
    st.subheader("Recall@10")
    st.bar_chart(eval_df.set_index("Strategy")["Recall@10"])

with col3:
    st.subheader("MRR")
    st.bar_chart(eval_df.set_index("Strategy")["MRR"])

st.divider()

# Summary table
st.subheader("Summary Table")
st.dataframe(
    eval_df.style.highlight_max(
        subset=["Recall@5", "Recall@10", "MRR"],
        color="lightgreen",
    ).highlight_min(
        subset=["Avg Latency (ms)"],
        color="lightgreen",
    ),
    use_container_width=True,
)

st.divider()

# Per-query breakdown
st.header("Per-Query Analysis")

per_query_df = get_mock_per_query_results()

# Pivot for comparison
pivot_df = per_query_df.pivot(
    index="Query",
    columns="Strategy",
    values="Recall@5",
)

st.dataframe(pivot_df, use_container_width=True)

st.divider()

# Strategy recommendation
st.header("Recommendation")

best_strategy = eval_df.loc[eval_df["Recall@10"].idxmax(), "Strategy"]
best_recall = eval_df["Recall@10"].max()

st.success(
    f"**Recommended Strategy: {best_strategy.title()}**\n\n"
    f"Based on the evaluation results, the **{best_strategy}** chunking strategy "
    f"achieves the highest Recall@10 of **{best_recall:.0%}**, indicating it retrieves "
    "the most relevant documents within the top 10 results."
)

st.markdown("""
### Strategy Descriptions

- **Fixed**: Splits documents into fixed-size chunks (512 tokens) with overlap.
  Simple and consistent, but may split semantic units.

- **Paragraph**: Splits on paragraph boundaries, preserving natural document structure.
  Better semantic coherence but variable chunk sizes.

- **Recursive**: Uses legal-specific separators (sections, paragraphs, sentences).
  Optimized for legal document structure.
""")
