"""
Training utilities and configuration
"""

from typing import Optional
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
)
from datasets import Dataset


def get_training_args(
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    eval_steps: int = 50,
    save_steps: int = 100,
    save_total_limit: int = 3,
) -> TrainingArguments:
    """
    Create training arguments configuration

    Args:
        output_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        logging_steps: Log every N steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        save_total_limit: Maximum number of checkpoints to keep

    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        gradient_checkpointing=True,
        fp16=False,
        bf16=False,
        logging_steps=logging_steps,
        report_to="none",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        seed=42,
    )


def create_trainer(
    model,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: TrainingArguments,
) -> Trainer:
    """
    Create a Trainer instance

    Args:
        model: The model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        training_args: Training arguments

    Returns:
        Trainer instance
    """
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Create and return trainer
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
