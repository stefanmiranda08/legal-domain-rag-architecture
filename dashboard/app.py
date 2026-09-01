"""Streamlit dashboard for Legal RAG observability."""

import streamlit as st

st.set_page_config(
    page_title="Legal RAG Dashboard",
    page_icon="⚖️",
    layout="wide",
)

st.title("Legal RAG Dashboard")
st.markdown("Observability dashboard for the Australian Legal Corpus RAG system.")

st.sidebar.title("Navigation")
st.sidebar.info(
    "Use the pages in the sidebar to explore different aspects of the RAG system."
)

# Main page content
st.header("System Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Documents", "232,560", help="Documents in the Australian Legal Corpus")

with col2:
    st.metric("Chunking Strategies", "3", help="Fixed, Paragraph, Recursive")

with col3:
    st.metric("Jurisdictions", "7", help="Commonwealth and state jurisdictions")

st.divider()

st.subheader("Quick Links")

st.markdown("""
- **Chunking Comparison**: Compare retrieval performance across different chunking strategies
- **Retrieval Metrics**: View latency, query volume, and filter usage statistics
- **Query Logs**: Browse recent queries and their results
""")

st.divider()

st.subheader("About")

st.markdown("""
This dashboard provides observability into the Legal RAG system, which uses
Retrieval-Augmented Generation to answer questions about Australian law.

**Data Source**: Open Australian Legal Corpus (232K documents)

**Components**:
- Vector search with Qdrant
- Metadata filtering by jurisdiction, document type, date
- LLM generation with OpenAI GPT
- Evaluation framework for strategy comparison
""")
