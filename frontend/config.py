"""
Configuration and initialization utilities
"""

import streamlit as st


def load_css() -> None:
    """Legacy function - CSS loading removed to use native Streamlit styling"""
    pass


def init_session_state() -> None:
    """Initialize session state variables"""
    if "training_started" not in st.session_state:
        st.session_state.training_started = False
    if "training_complete" not in st.session_state:
        st.session_state.training_complete = False
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "device" not in st.session_state:
        st.session_state.device = None
    if "trainer" not in st.session_state:
        st.session_state.trainer = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
