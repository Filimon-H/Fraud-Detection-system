"""
Pipeline builders for fraud detection models.

This module provides functions to build sklearn/imblearn pipelines
with preprocessing and optional SMOTE resampling.
"""

from typing import List, Literal, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# Default feature lists for e-commerce fraud data
DEFAULT_NUMERIC_FEATURES = [
    "purchase_value",
    "age",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "time_since_signup",
    "tx_count_user_id_1h",
    "tx_count_user_id_24h",
    "user_total_transactions",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "source",
    "browser",
    "sex",
    "country",
]


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Build a preprocessing transformer for mixed feature types.

    Parameters
    ----------
    numeric_features : List[str]
        List of numeric column names.
    categorical_features : List[str]
        List of categorical column names.

    Returns
    -------
    ColumnTransformer
        Fitted transformer with scaling and one-hot encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    return preprocessor


def build_fraud_pipeline(
    model_type: Literal["logistic", "random_forest", "gradient_boosting"] = "logistic",
    use_smote: bool = True,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    random_state: int = 42,
    **model_params,
) -> ImbPipeline:
    """
    Build a complete pipeline for e-commerce fraud detection.

    Parameters
    ----------
    model_type : str
        Type of classifier: 'logistic', 'random_forest', or 'gradient_boosting'.
    use_smote : bool
        Whether to include SMOTE resampling in the pipeline.
    numeric_features : List[str], optional
        Numeric feature names. Uses defaults if not provided.
    categorical_features : List[str], optional
        Categorical feature names. Uses defaults if not provided.
    random_state : int
        Random state for reproducibility.
    **model_params
        Additional parameters passed to the classifier.

    Returns
    -------
    ImbPipeline
        Complete pipeline with preprocessing, optional SMOTE, and classifier.
    """
    if numeric_features is None:
        numeric_features = DEFAULT_NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES

    # Build preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Build classifier
    if model_type == "logistic":
        default_params = {"class_weight": "balanced", "max_iter": 1000, "random_state": random_state}
        default_params.update(model_params)
        classifier = LogisticRegression(**default_params)

    elif model_type == "random_forest":
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced_subsample",
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(model_params)
        classifier = RandomForestClassifier(**default_params)

    elif model_type == "gradient_boosting":
        default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": random_state,
        }
        default_params.update(model_params)
        classifier = GradientBoostingClassifier(**default_params)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Build pipeline
    steps = [("preprocessor", preprocessor)]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))

    steps.append(("classifier", classifier))

    return ImbPipeline(steps)


def build_creditcard_pipeline(
    model_type: Literal["logistic", "random_forest", "gradient_boosting"] = "logistic",
    use_smote: bool = True,
    random_state: int = 42,
    **model_params,
) -> ImbPipeline:
    """
    Build a pipeline for credit card fraud detection.

    Credit card data has different features (V1-V28, Time, Amount).
    All features are numeric.

    Parameters
    ----------
    model_type : str
        Type of classifier.
    use_smote : bool
        Whether to include SMOTE.
    random_state : int
        Random state for reproducibility.
    **model_params
        Additional parameters passed to the classifier.

    Returns
    -------
    ImbPipeline
        Complete pipeline with scaling, optional SMOTE, and classifier.
    """
    # Build classifier
    if model_type == "logistic":
        default_params = {"class_weight": "balanced", "max_iter": 1000, "random_state": random_state}
        default_params.update(model_params)
        classifier = LogisticRegression(**default_params)

    elif model_type == "random_forest":
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "class_weight": "balanced_subsample",
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_params.update(model_params)
        classifier = RandomForestClassifier(**default_params)

    elif model_type == "gradient_boosting":
        default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": random_state,
        }
        default_params.update(model_params)
        classifier = GradientBoostingClassifier(**default_params)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Build pipeline (all numeric, just scale)
    steps = [("scaler", StandardScaler())]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))

    steps.append(("classifier", classifier))

    return ImbPipeline(steps)


def get_model_name(model_type: str, use_smote: bool) -> str:
    """
    Generate a descriptive model name.

    Parameters
    ----------
    model_type : str
        Type of classifier.
    use_smote : bool
        Whether SMOTE is used.

    Returns
    -------
    str
        Descriptive model name.
    """
    names = {
        "logistic": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }
    base = names.get(model_type, model_type)
    suffix = " + SMOTE" if use_smote else ""
    return base + suffix
