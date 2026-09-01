"""Query logs page."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random

st.set_page_config(
    page_title="Query Logs",
    page_icon="",
    layout="wide",
)

st.title("Query Logs")
st.markdown("Browse recent queries and their results.")


def generate_mock_query_logs(n: int = 50) -> pd.DataFrame:
    """Generate mock query log data."""
    queries = [
        "What constitutes negligence under Australian law?",
        "Define reasonable care in tort law",
        "Explain vicarious liability in employment",
        "What is the duty of care owed by employers?",
        "Define causation in negligence cases",
        "What are the elements of a valid contract?",
        "Explain the doctrine of consideration",
        "What is promissory estoppel?",
        "Define breach of contract remedies",
        "Explain statutory interpretation principles",
        "What is the doctrine of precedent?",
        "Define administrative law review grounds",
        "Explain natural justice requirements",
        "What is judicial review in Australia?",
        "Define constitutional validity tests",
    ]

    jurisdictions = [
        "commonwealth",
        "new_south_wales",
        "victoria",
        "queensland",
        "western_australia",
        None,
    ]

    strategies = ["fixed", "paragraph", "recursive"]

    data = []
    base_time = datetime.now()

    for i in range(n):
        query_time = base_time - timedelta(
            hours=random.randint(0, 72),
            minutes=random.randint(0, 59),
        )

        latency = random.randint(80, 250)
        chunks_retrieved = random.randint(5, 10)

        data.append({
            "id": f"q_{i+1:04d}",
            "timestamp": query_time,
            "query": random.choice(queries),
            "jurisdiction": random.choice(jurisdictions),
            "strategy": random.choice(strategies),
            "chunks_retrieved": chunks_retrieved,
            "latency_ms": latency,
            "tokens_used": random.randint(400, 800),
        })

    df = pd.DataFrame(data)
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


# Sidebar filters
st.sidebar.header("Filters")

search_query = st.sidebar.text_input(
    "Search Queries",
    placeholder="Enter search term...",
)

jurisdiction_filter = st.sidebar.multiselect(
    "Jurisdiction",
    ["commonwealth", "new_south_wales", "victoria", "queensland", "western_australia"],
    default=[],
)

strategy_filter = st.sidebar.multiselect(
    "Chunking Strategy",
    ["fixed", "paragraph", "recursive"],
    default=[],
)

time_filter = st.sidebar.selectbox(
    "Time Range",
    ["Last hour", "Last 24 hours", "Last 7 days", "All time"],
    index=1,
)

# Load data
logs_df = generate_mock_query_logs(100)

# Apply filters
if search_query:
    logs_df = logs_df[
        logs_df["query"].str.contains(search_query, case=False, na=False)
    ]

if jurisdiction_filter:
    logs_df = logs_df[logs_df["jurisdiction"].isin(jurisdiction_filter)]

if strategy_filter:
    logs_df = logs_df[logs_df["strategy"].isin(strategy_filter)]

# Summary metrics
st.header("Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Queries", len(logs_df))

with col2:
    avg_latency = logs_df["latency_ms"].mean()
    st.metric("Avg Latency", f"{avg_latency:.0f}ms")

with col3:
    avg_chunks = logs_df["chunks_retrieved"].mean()
    st.metric("Avg Chunks Retrieved", f"{avg_chunks:.1f}")

with col4:
    total_tokens = logs_df["tokens_used"].sum()
    st.metric("Total Tokens", f"{total_tokens:,}")

st.divider()

# Query logs table
st.header("Recent Queries")

# Display columns
display_cols = [
    "timestamp",
    "query",
    "jurisdiction",
    "strategy",
    "chunks_retrieved",
    "latency_ms",
]

# Format for display
display_df = logs_df[display_cols].copy()
display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
display_df["jurisdiction"] = display_df["jurisdiction"].fillna("(none)")
display_df.columns = [
    "Timestamp",
    "Query",
    "Jurisdiction",
    "Strategy",
    "Chunks",
    "Latency (ms)",
]

st.dataframe(
    display_df,
    use_container_width=True,
    height=400,
)

st.divider()

# Query detail view
st.header("Query Details")

selected_query = st.selectbox(
    "Select a query to view details",
    logs_df["id"].tolist(),
    format_func=lambda x: f"{x}: {logs_df[logs_df['id'] == x]['query'].values[0][:50]}...",
)

if selected_query:
    query_row = logs_df[logs_df["id"] == selected_query].iloc[0]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Query Text")
        st.text(query_row["query"])

        st.subheader("Retrieved Chunks")

        # Mock chunk data
        for i in range(min(3, query_row["chunks_retrieved"])):
            with st.expander(f"Chunk {i+1} (Score: {0.95 - i*0.05:.2f})"):
                st.markdown(f"""
**Document ID**: doc_{random.randint(1000, 9999)}

**Citation**: Sample Case [{2020 + i}] HCA {random.randint(1, 50)}

**Excerpt**:
> This is a sample excerpt from the retrieved chunk. In a real
> implementation, this would contain the actual text from the
> legal document that was retrieved based on semantic similarity
> to the query.
                """)

    with col2:
        st.subheader("Metadata")
        st.json({
            "query_id": query_row["id"],
            "timestamp": str(query_row["timestamp"]),
            "jurisdiction": query_row["jurisdiction"],
            "strategy": query_row["strategy"],
            "chunks_retrieved": int(query_row["chunks_retrieved"]),
            "latency_ms": int(query_row["latency_ms"]),
            "tokens_used": int(query_row["tokens_used"]),
        })

st.divider()

# Export option
st.header("Export")

col1, col2 = st.columns(2)

with col1:
    csv_data = logs_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="query_logs.csv",
        mime="text/csv",
    )

with col2:
    json_data = logs_df.to_json(orient="records", date_format="iso")
    st.download_button(
        label="Download JSON",
        data=json_data,
        file_name="query_logs.json",
        mime="application/json",
    )
