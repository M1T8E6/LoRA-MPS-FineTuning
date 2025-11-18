"""
Sidebar configuration UI components
"""

import streamlit as st
import torch
from backend import check_mps_availability


def render_sidebar():
    """
    Render the sidebar with device information
    """
    st.sidebar.header("⚙️ System Info")

    # Device check
    device, _, status_msg = check_mps_availability()
    st.session_state.device = device

    st.sidebar.info(status_msg)
    st.sidebar.subheader(f"PyTorch: {torch.__version__}", divider=True)

    # Navigation hint
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    ### 📖 Guide
    1. **Setup**: Configure model and dataset
    2. **Training**: Start the training process
    3. **Testing**: Test your fine-tuned model
    4. **Export**: Save your model
    """
    )
