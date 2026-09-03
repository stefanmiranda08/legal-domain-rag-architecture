"""Retrieval metrics page."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from styles import apply_global_styles

st.set_page_config(
    page_title="Retrieval Metrics",
    page_icon="",
    layout="wide",
)

apply_global_styles()

st.title("Retrieval Metrics")
st.markdown("Monitor query latency, volume, and filter usage statistics.")


def generate_time_series_data(days: int = 7) -> pd.DataFrame:
    """Generate mock time series data for queries."""
    dates = pd.date_range(end=datetime.now(), periods=days * 24, freq="h")

    data = []
    for dt in dates:
        hour = dt.hour
        base_volume = 10 + (5 if 9 <= hour <= 17 else 0)
        volume = base_volume + random.randint(-3, 5)
        avg_latency = 120 + random.randint(-20, 40)

        data.append({
            "timestamp": dt,
            "query_count": max(0, volume),
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": avg_latency + random.randint(30, 80),
        })

    return pd.DataFrame(data)


def get_filter_usage_data() -> pd.DataFrame:
    """Generate mock filter usage statistics."""
    return pd.DataFrame({
        "Filter Type": [
            "jurisdiction",
            "document_type",
            "date_range",
            "no_filter",
        ],
        "Usage Count": [1250, 890, 340, 2100],
        "Percentage": [27.3, 19.4, 7.4, 45.9],
    })


def get_jurisdiction_breakdown() -> pd.DataFrame:
    """Generate jurisdiction usage breakdown."""
    return pd.DataFrame({
        "Jurisdiction": [
            "Commonwealth",
            "New South Wales",
            "Victoria",
            "Queensland",
            "Western Australia",
            "South Australia",
            "Tasmania",
        ],
        "Query Count": [450, 380, 290, 180, 120, 80, 50],
    })


# Sidebar controls
st.sidebar.header("Time Range")

time_range = st.sidebar.selectbox(
    "Select Range",
    ["Last 24 hours", "Last 7 days", "Last 30 days"],
    index=1,
)

refresh = st.sidebar.button("Refresh Data")

# Main metrics
st.header("Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Queries",
        "4,580",
        delta="+12%",
        help="Total queries in selected time range",
    )

with col2:
    st.metric(
        "Avg Latency",
        "142ms",
        delta="-8ms",
        delta_color="inverse",
        help="Average query latency",
    )

with col3:
    st.metric(
        "P95 Latency",
        "215ms",
        delta="-12ms",
        delta_color="inverse",
        help="95th percentile latency",
    )

with col4:
    st.metric(
        "Error Rate",
        "0.2%",
        delta="-0.1%",
        delta_color="inverse",
        help="Query error rate",
    )

st.divider()

# Time series charts
st.header("Query Volume Over Time")

time_df = generate_time_series_data(7)

# Resample to daily for cleaner visualization
daily_df = time_df.set_index("timestamp").resample("D").agg({
    "query_count": "sum",
    "avg_latency_ms": "mean",
    "p95_latency_ms": "mean",
}).reset_index()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Query Volume")
    st.line_chart(daily_df.set_index("timestamp")["query_count"])

with col2:
    st.subheader("Latency Trends")
    latency_chart_data = daily_df.set_index("timestamp")[["avg_latency_ms", "p95_latency_ms"]]
    latency_chart_data.columns = ["Average", "P95"]
    st.line_chart(latency_chart_data)

st.divider()

# Filter usage
st.header("Filter Usage Statistics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Filter Types")
    filter_df = get_filter_usage_data()
    st.bar_chart(filter_df.set_index("Filter Type")["Usage Count"])

with col2:
    st.subheader("Usage Breakdown")
    st.dataframe(filter_df, use_container_width=True)

st.divider()

# Jurisdiction breakdown
st.header("Queries by Jurisdiction")

jurisdiction_df = get_jurisdiction_breakdown()

col1, col2 = st.columns([2, 1])

with col1:
    st.bar_chart(jurisdiction_df.set_index("Jurisdiction")["Query Count"])

with col2:
    st.dataframe(jurisdiction_df, use_container_width=True)

st.divider()

# Latency distribution
st.header("Latency Distribution")

st.markdown("""
| Percentile | Latency |
|------------|---------|
| P50 | 125ms |
| P75 | 165ms |
| P90 | 195ms |
| P95 | 215ms |
| P99 | 285ms |
""")

st.info(
    "Latency targets: P50 < 150ms, P95 < 250ms, P99 < 500ms. "
    "Current performance meets all targets."
)
