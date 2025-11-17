"""
Model loading and device utilities
"""

from typing import Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def check_mps_availability() -> Tuple[torch.device, bool, str]:
    """
    Check if MPS is available and return device info

    Returns:
        Tuple containing:
        - torch.device: The device to use (mps or cpu)
        - bool: Whether MPS is available
        - str: Status message
    """
    mps_is_available = torch.backends.mps.is_available()

    if mps_is_available:
        dev = torch.device("mps")
        msg = "✅ MPS (Apple Silicon) available"
    else:
        dev = torch.device("cpu")
        msg = "⚠️ MPS not available, using CPU (slower)"

    return dev, mps_is_available, msg


def load_model_and_tokenizer(
    model_id: str, target_device: torch.device
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load model and tokenizer from HuggingFace

    Args:
        model_id: HuggingFace model identifier
        target_device: Device to load the model on

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    loaded_tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )

    # Set pad token if not exists
    if loaded_tokenizer.pad_token is None:
        loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

    # Load model
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": target_device},
        trust_remote_code=True,
    )

    return loaded_model, loaded_tokenizer
