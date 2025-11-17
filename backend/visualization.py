"""
Visualization utilities for training metrics
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_training_metrics(training_history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create interactive training metrics plot

    Args:
        training_history: List of training log entries

    Returns:
        Plotly Figure object
    """
    # Extract losses
    losses_train = [entry["loss"] for entry in training_history if "loss" in entry]
    losses_eval = [
        entry["eval_loss"] for entry in training_history if "eval_loss" in entry
    ]

    # Create subplots
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=("Training Loss", "Validation Loss")
    )

    # Add training loss trace
    if losses_train:
        figure.add_trace(
            go.Scatter(
                y=losses_train,
                mode="lines",
                name="Training Loss",
                line=dict(color="#667eea", width=2),
            ),
            row=1,
            col=1,
        )

    # Add validation loss trace
    if losses_eval:
        figure.add_trace(
            go.Scatter(
                y=losses_eval,
                mode="lines",
                name="Validation Loss",
                line=dict(color="#f093fb", width=2),
            ),
            row=1,
            col=2,
        )

    # Update layout
    figure.update_layout(height=400, showlegend=True, title_text="Training Metrics")

    return figure
