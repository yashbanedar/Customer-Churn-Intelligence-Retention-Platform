"""
Scikit-Learn Preprocessing and Estimator Pipeline Factory.
Ensures zero data leakage by bundling imputation, encoding, and scaling
strictly inside scikit-learn Pipeline and ColumnTransformer objects.
"""

from typing import List, Literal, Optional, Tuple, Union
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# Numeric continuous features to be imputed and scaled
NUMERIC_FEATURES: List[str] = [
    "tenure",
    "monthly_charges",
    "total_charges",
]

# Categorical features to be imputed and one-hot encoded
CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
]

ALL_FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs the ColumnTransformer for leakage-safe preprocessing:
    - Numeric: Median imputation + StandardScaler
    - Categorical: Most-frequent imputation + OneHotEncoder (ignore unknown levels)

    Returns:
        Configured ColumnTransformer instance.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # Drops identifier columns like customer_id automatically
    )

    return preprocessor


def build_model_pipeline(
    model_type: Literal["logistic_regression", "random_forest", "xgboost"] = "logistic_regression",
    scale_pos_weight: float = 2.77,
    random_state: int = 42,
    **model_kwargs,
) -> Pipeline:
    """
    Constructs a complete scikit-learn Pipeline combining the preprocessing transformer
    and the specified classifier.

    Args:
        model_type: One of 'logistic_regression', 'random_forest', 'xgboost'.
        scale_pos_weight: Imbalance ratio for XGBoost (defaults to ~2.77 based on EDA).
        random_state: Seed for reproducibility.
        model_kwargs: Optional hyperparameter overrides passed to the estimator.

    Returns:
        Configured scikit-learn Pipeline instance.
    """
    preprocessor = build_preprocessor()

    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state,
            **model_kwargs,
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            **model_kwargs,
        )
    elif model_type == "xgboost":
        classifier = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            **model_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Must be one of: 'logistic_regression', 'random_forest', 'xgboost'."
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline
