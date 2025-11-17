"""
Configuration and initialization utilities
"""

import os
import streamlit as st


def load_css() -> None:
    """Load custom CSS from external file"""
    css_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "style", "main.css"
    )
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ CSS file not found!")


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
