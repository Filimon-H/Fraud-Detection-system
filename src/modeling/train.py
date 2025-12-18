"""
Training and evaluation utilities for fraud detection models.

This module provides functions to train models, perform cross-validation,
and save/load model artifacts.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import clone

from src.modeling.metrics import compute_classification_metrics, find_best_threshold


def train_and_evaluate(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    optimize_threshold: bool = False,
    threshold_metric: str = "f1",
) -> Tuple[Any, Dict[str, float], float]:
    """
    Train a pipeline and evaluate on test set.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn/imblearn pipeline to train.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    threshold : float
        Classification threshold for predictions.
    optimize_threshold : bool
        If True, find optimal threshold on training set predictions.
    threshold_metric : str
        Metric to optimize threshold for ('f1', 'precision', 'recall').

    Returns
    -------
    Tuple[Pipeline, Dict, float]
        - Trained pipeline
        - Dictionary of evaluation metrics
        - Final threshold used
    """
    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Get probabilities
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Optionally optimize threshold
    if optimize_threshold:
        # Use cross-validation on training set to find threshold
        y_train_proba = cross_val_predict(
            clone(pipeline), X_train, y_train,
            cv=3, method="predict_proba"
        )[:, 1]
        threshold, _ = find_best_threshold(y_train.values, y_train_proba, metric=threshold_metric)

    # Make predictions with threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Compute metrics
    metrics = compute_classification_metrics(y_test.values, y_pred, y_proba)
    metrics["threshold"] = threshold

    return pipeline, metrics, threshold


def cross_validate_model(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Perform stratified cross-validation and return metrics.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn/imblearn pipeline to evaluate.
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.
    cv : int
        Number of folds.
    random_state : int
        Random state for fold generation.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - fold_metrics: List of metrics per fold
        - mean_metrics: Mean of each metric across folds
        - std_metrics: Std of each metric across folds
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # Clone and fit pipeline
        pipeline_clone = clone(pipeline)
        pipeline_clone.fit(X_train_fold, y_train_fold)

        # Predict
        y_proba = pipeline_clone.predict_proba(X_val_fold)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        # Compute metrics
        metrics = compute_classification_metrics(y_val_fold.values, y_pred, y_proba)
        metrics["fold"] = fold_idx + 1
        fold_metrics.append(metrics)

    # Aggregate metrics
    metrics_df = pd.DataFrame(fold_metrics)
    numeric_cols = ["precision", "recall", "f1", "auc_pr", "roc_auc"]
    numeric_cols = [c for c in numeric_cols if c in metrics_df.columns]

    mean_metrics = metrics_df[numeric_cols].mean().to_dict()
    std_metrics = metrics_df[numeric_cols].std().to_dict()

    return {
        "fold_metrics": fold_metrics,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
    }


def save_model(
    pipeline: Any,
    model_name: str,
    metrics: Dict[str, float],
    output_dir: Path,
) -> Path:
    """
    Save a trained model and its metrics.

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline to save.
    model_name : str
        Name for the model file.
    metrics : Dict
        Evaluation metrics to save alongside.
    output_dir : Path
        Directory to save to.

    Returns
    -------
    Path
        Path to saved model file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean model name for filename
    clean_name = model_name.lower().replace(" ", "_").replace("+", "")

    # Save model
    model_path = output_dir / f"{clean_name}.joblib"
    joblib.dump(pipeline, model_path)

    # Save metrics
    metrics_path = output_dir / f"{clean_name}_metrics.joblib"
    joblib.dump(metrics, metrics_path)

    return model_path


def load_model(model_path: Path) -> Any:
    """
    Load a saved model.

    Parameters
    ----------
    model_path : Path
        Path to the saved model file.

    Returns
    -------
    Pipeline
        Loaded pipeline.
    """
    return joblib.load(model_path)


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison DataFrame from multiple model results.

    Parameters
    ----------
    results : Dict[str, Dict]
        Dictionary mapping model names to their metrics dicts.

    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns.
    """
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        for key in ["auc_pr", "roc_auc", "f1", "precision", "recall", "threshold"]:
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("model")

    # Sort by AUC-PR (primary metric) descending
    if "auc_pr" in df.columns:
        df = df.sort_values("auc_pr", ascending=False)

    return df
