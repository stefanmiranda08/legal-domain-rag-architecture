"""Shared styles for the Streamlit dashboard."""

import streamlit as st


def apply_global_styles():
    """Apply global CSS styles to reduce text size and improve readability."""
    st.markdown("""
<style>
    /* Reduce base font size */
    html, body, [class*="css"] {
        font-size: 14px;
    }

    /* Reduce font size in chat messages */
    .stChatMessage {
        font-size: 0.9rem;
    }

    /* Reduce font size in main content area */
    .stMarkdown {
        font-size: 0.9rem;
    }

    /* Reduce font size in expanders */
    .streamlit-expanderContent {
        font-size: 0.85rem;
    }

    /* Reduce font size in dataframes and tables */
    .stDataFrame {
        font-size: 0.85rem;
    }

    /* Reduce font size in metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }

    /* Adjust line height for readability */
    .stChatMessage p, .stMarkdown p {
        line-height: 1.5;
    }

    /* Reduce sidebar text */
    .stSidebar {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)
