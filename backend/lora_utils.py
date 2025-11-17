"""
LoRA configuration and application utilities
"""

from typing import List, Optional, Set, Union
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import PreTrainedModel


def find_target_modules(
    base_model: Union[PreTrainedModel, PeftModel],
    exclude_names: Optional[Set[str]] = None,
) -> List[str]:
    """
    Identify candidate module names to apply LoRA

    Args:
        base_model: The model to inspect
        exclude_names: Set of module names to exclude

    Returns:
        List of module names suitable for LoRA
    """
    if exclude_names is None:
        exclude_names = {"lm_head", "embed_tokens", "wte", "wpe", "ln_f"}

    found_modules: Set[str] = set()
    for name, module in base_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_name = name.split(".")[-1]
            if module_name:
                found_modules.add(module_name)

    return list(found_modules - exclude_names)


def create_lora_config(
    r: int,
    alpha: int,
    dropout: float,
    modules_to_target: List[str]
) -> LoraConfig:
    """
    Create LoRA configuration

    Args:
        r: LoRA rank
        alpha: LoRA alpha (scaling factor)
        dropout: Dropout probability
        modules_to_target: List of module names to apply LoRA to

    Returns:
        LoraConfig object
    """
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=modules_to_target,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def apply_lora_to_model(
    model: PreTrainedModel,
    lora_config: LoraConfig,
    enable_gradient_checkpointing: bool = True,
) -> PeftModel:
    """
    Apply LoRA to a model and enable optimizations

    Args:
        model: The base model
        lora_config: LoRA configuration
        enable_gradient_checkpointing: Whether to enable gradient checkpointing

    Returns:
        Model with LoRA applied
    """
    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)

    # Enable optimizations
    if enable_gradient_checkpointing:
        peft_model.gradient_checkpointing_enable()  # type: ignore

    peft_model.enable_input_require_grads()  # type: ignore

    if hasattr(peft_model, "config"):
        peft_model.config.use_cache = False  # type: ignore

    return peft_model
