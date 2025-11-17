"""
Sidebar configuration UI components
"""

from typing import Tuple
import streamlit as st
import torch
from backend import check_mps_availability


def render_sidebar():
    """
    Render the sidebar with all configuration options

    Returns:
        Tuple containing all configuration parameters:
        (model_name, dataset_name, num_samples, max_length, lora_r, lora_alpha,
         lora_dropout, num_epochs, batch_size, gradient_accumulation, learning_rate, output_name)
    """
    st.sidebar.header("⚙️ Configuration")

    # Device check
    device, _, status_msg = check_mps_availability()
    st.session_state.device = device

    st.sidebar.info(status_msg)
    st.sidebar.subheader(f"PyTorch: {torch.__version__}", divider=True)

    # Model selection
    st.sidebar.subheader("🤖 Model")
    model_selection_type = st.sidebar.radio(
        "Selection mode",
        ["Preset", "Custom"],
        horizontal=True,
        help="Choose a preset model or enter a custom path",
    )

    if model_selection_type == "Preset":
        model_name = st.sidebar.selectbox(
            "Select base model",
            [
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
            ],
            help="Smaller models (1B) are faster but less capable",
        )
    else:
        model_name = st.sidebar.text_input(
            "HuggingFace model path",
            value="",
            help="Enter the model path (e.g., 'meta-llama/Llama-3.2-1B' or 'gpt2')",
            placeholder="username/model-name",
        )

    # Dataset selection
    st.sidebar.subheader("📊 Dataset")
    dataset_selection_type = st.sidebar.radio(
        "Dataset selection mode",
        ["Preset", "Custom"],
        horizontal=True,
        help="Choose a preset dataset or enter a custom path",
        key="dataset_radio",
    )

    if dataset_selection_type == "Preset":
        dataset_name = st.sidebar.selectbox(
            "Select dataset",
            ["imdb", "wikitext-2-raw-v1", "tiny_shakespeare"],
            help="IMDB: movie reviews, WikiText: Wikipedia text",
        )
    else:
        dataset_name = st.sidebar.text_input(
            "HuggingFace dataset path",
            value="",
            help="Enter the dataset path (e.g., 'squad', 'glue/mrpc', 'wikipedia')",
            placeholder="username/dataset-name or dataset-name",
        )

    num_samples = st.sidebar.slider(
        "Number of samples",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="More samples = longer training but better results",
    )

    max_length = st.sidebar.slider(
        "Max sequence length",
        min_value=128,
        max_value=1024,
        value=512,
        step=128,
        help="Longer sequences require more memory",
    )

    # LoRA configuration
    st.sidebar.subheader("🔧 LoRA Parameters")
    lora_r = st.sidebar.slider(
        "Rank (r)",
        min_value=4,
        max_value=64,
        value=16,
        step=4,
        help="LoRA matrix dimension. Higher = more parameters",
    )

    lora_alpha = st.sidebar.slider(
        "Alpha",
        min_value=8,
        max_value=128,
        value=32,
        step=8,
        help="Scaling factor. Generally alpha = 2 × r",
    )

    lora_dropout = st.sidebar.slider(
        "Dropout",
        min_value=0.0,
        max_value=0.2,
        value=0.05,
        step=0.05,
        help="Dropout for regularization",
    )

    # Training configuration
    st.sidebar.subheader("🎯 Training")
    num_epochs = st.sidebar.slider(
        "Epochs",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of complete passes through the dataset",
    )

    batch_size = st.sidebar.slider(
        "Batch size",
        min_value=1,
        max_value=16,
        value=4,
        step=1,
        help="Reduce if you have memory issues",
    )

    gradient_accumulation = st.sidebar.slider(
        "Gradient accumulation steps",
        min_value=1,
        max_value=16,
        value=4,
        step=1,
        help="Simulates larger batch sizes",
    )

    learning_rate = st.sidebar.select_slider(
        "Learning rate",
        options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
        value=2e-4,
        help="Learning speed",
    )

    # Output name
    output_name = st.sidebar.text_input(
        "Output model name",
        value="my-lora-model",
        help="Name to save the fine-tuned model",
    )

    return (
        model_name,
        dataset_name,
        num_samples,
        max_length,
        lora_r,
        lora_alpha,
        lora_dropout,
        num_epochs,
        batch_size,
        gradient_accumulation,
        learning_rate,
        output_name,
    )
