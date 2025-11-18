"""
🚀 LoRA Fine-Tuning Interface
User-friendly Streamlit app for fine-tuning LLMs with LoRA on Mac (MPS)
"""

import streamlit as st

# Frontend imports
from frontend import (
    load_css,
    init_session_state,
    render_sidebar,
    render_setup_tab,
    render_training_tab,
    render_testing_tab,
    render_export_tab,
)

# Set page config
st.set_page_config(
    page_title="Just a Fine-Tuning Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize
load_css()
init_session_state()

# =============================================================================
# SIDEBAR - System Info
# =============================================================================

render_sidebar()

# =============================================================================
# MAIN AREA
# =============================================================================

st.title("🚀 Just a Fine-Tuning Studio")
st.subheader("Fine-tuning LLMs on Mac with LoRA and MPS")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Setup", "🎓 Training", "🧪 Testing", "💾 Export"])

# =============================================================================
# RENDER TABS
# =============================================================================

with tab1:
    render_setup_tab(device=st.session_state.device)

with tab2:
    render_training_tab()

with tab3:
    render_testing_tab(device=st.session_state.device)

with tab4:
    render_export_tab()
# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚀 Just a Fine-Tuning Studio | Made with ❤️ by <a href="https://justanotherai.company">Just Another AI Company</a></p>
        <p>Powered by 🤗 Transformers, PEFT, and Apple MPS</p>
    </div>
    """,
    unsafe_allow_html=True,
)
