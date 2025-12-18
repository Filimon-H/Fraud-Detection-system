"""
Modeling utilities for fraud detection.

This package provides functions for building pipelines, training models,
evaluating performance, and selecting optimal thresholds.
"""

from src.modeling.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
)
from src.modeling.train import train_and_evaluate, cross_validate_model
from src.modeling.pipelines import build_fraud_pipeline, build_creditcard_pipeline

__all__ = [
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "train_and_evaluate",
    "cross_validate_model",
    "build_fraud_pipeline",
    "build_creditcard_pipeline",
]
