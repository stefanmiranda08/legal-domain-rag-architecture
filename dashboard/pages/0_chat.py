"""Chat interface for the Legal RAG assistant."""

import streamlit as st
import httpx
import os
import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from styles import apply_global_styles

st.set_page_config(
    page_title="Legal Assistant",
    page_icon="",
    layout="wide",
)

apply_global_styles()

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Australian Legal Research Assistant")
st.markdown(
    "Ask questions about Australian law. This assistant uses RAG over the "
    "Open Australian Legal Corpus (232K documents)."
)

# Sidebar filters
st.sidebar.header("Search Filters")

jurisdiction = st.sidebar.selectbox(
    "Jurisdiction",
    [
        None,
        "commonwealth",
        "new_south_wales",
        "victoria",
        "queensland",
        "western_australia",
        "south_australia",
        "tasmania",
    ],
    format_func=lambda x: "All Jurisdictions" if x is None else x.replace("_", " ").title(),
)

document_type = st.sidebar.selectbox(
    "Document Type",
    [None, "decision", "legislation", "secondary_material"],
    format_func=lambda x: "All Types" if x is None else x.replace("_", " ").title(),
)

chunking_strategy = st.sidebar.selectbox(
    "Chunking Strategy",
    ["recursive", "fixed", "paragraph"],
    help="Strategy used to split documents for retrieval",
)

top_k = st.sidebar.slider(
    "Number of Sources",
    min_value=3,
    max_value=20,
    value=10,
    help="Number of document chunks to retrieve",
)

st.sidebar.divider()
st.sidebar.markdown("**About**")
st.sidebar.markdown(
    "This assistant searches through Australian legal documents "
    "including court decisions, legislation, and secondary materials."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("View Sources"):
                for i, citation in enumerate(message["citations"], 1):
                    st.markdown(f"**{i}. {citation['citation']}**")
                    st.markdown(f"> {citation['excerpt']}")
                    st.caption(f"Relevance: {citation['relevance_score']:.0%}")

# Chat input
if prompt := st.chat_input("Ask a legal question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build request
    request_body = {
        "query": prompt,
        "top_k": top_k,
        "chunking_strategy": chunking_strategy,
    }

    # Add filters if specified
    filters = {}
    if jurisdiction:
        filters["jurisdiction"] = jurisdiction
    if document_type:
        filters["document_type"] = document_type
    if filters:
        request_body["filters"] = filters

    # Query API
    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents..."):
            try:
                response = httpx.post(
                    f"{API_URL}/query",
                    json=request_body,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "No answer generated.")
                citations = data.get("citations", [])

                # Display answer
                st.markdown(answer)

                # Display citations
                if citations:
                    with st.expander("View Sources"):
                        for i, citation in enumerate(citations, 1):
                            st.markdown(f"**{i}. {citation['citation']}**")
                            st.markdown(f"> {citation['excerpt']}")
                            st.caption(f"Relevance: {citation['relevance_score']:.0%}")

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                })

            except httpx.ConnectError:
                error_msg = "Could not connect to the API. Please ensure the backend is running."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
            except httpx.HTTPStatusError as e:
                error_msg = f"API error: {e.response.status_code}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

# Clear chat button
if st.session_state.messages:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
