"""
Explainability utilities for fraud detection models.

This package provides functions for SHAP-based model explanations,
including global and local interpretability plots.
"""

from src.explainability.shap_utils import (
    get_feature_names_from_pipeline,
    transform_for_explanation,
    create_tree_explainer,
    compute_shap_values,
    plot_shap_summary,
    plot_shap_bar,
    plot_shap_dependence,
    plot_shap_waterfall,
    get_example_cases,
)

__all__ = [
    "get_feature_names_from_pipeline",
    "transform_for_explanation",
    "create_tree_explainer",
    "compute_shap_values",
    "plot_shap_summary",
    "plot_shap_bar",
    "plot_shap_dependence",
    "plot_shap_waterfall",
    "get_example_cases",
]
