"""
Dataset loading and preprocessing utilities
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, Dataset, DatasetDict


def prepare_dataset(
    ds_name: str,
    n_samples: int,
    tok: PreTrainedTokenizerBase,
    seq_max_length: int
) -> Tuple[Dataset, Dataset, int]:
    """
    Load and tokenize dataset from HuggingFace

    Args:
        ds_name: Dataset name/path on HuggingFace
        n_samples: Number of samples to load
        tok: Tokenizer to use
        seq_max_length: Maximum sequence length

    Returns:
        Tuple of (train_dataset, eval_dataset, total_samples)
    """
    # Load dataset
    dataset = load_dataset(ds_name, split=f"train[:{n_samples}]")

    def tokenize_function(examples: Dict[str, Any]) -> Union[Dict[str, Any], Any]:
        """Tokenize a batch of examples"""
        return tok(
            examples["text"],
            truncation=True,
            max_length=seq_max_length,
            padding="max_length",
            return_tensors=None,
        )

    # Get column names - handle different dataset types
    columns_to_remove: Optional[List[str]] = None
    if hasattr(dataset, "column_names"):
        cols = dataset.column_names  # type: ignore
        # Convert to list if it's a list/tuple
        if isinstance(cols, (list, tuple)):
            columns_to_remove = list(cols)

    # Tokenize dataset
    tokenized_dataset = dataset.map(  # type: ignore
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
    )

    # Split into train/eval
    if isinstance(tokenized_dataset, Dataset):
        split_ds = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    elif isinstance(tokenized_dataset, DatasetDict):
        # Already split
        split_ds = tokenized_dataset
    else:
        # Fallback: try to call train_test_split
        split_ds = tokenized_dataset.train_test_split(
            test_size=0.1, seed=42
        )  # type: ignore

    return split_ds["train"], split_ds["test"], n_samples  # type: ignore
