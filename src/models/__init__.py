"""
Leakage-safe machine learning pipelines, probability calibration, threshold selection, and SHAP explainability.
"""

from src.models.evaluator import (
    ModelEvaluationMetrics,
    compute_evaluation_metrics,
    plot_model_comparisons,
)
from src.models.pipeline import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_model_pipeline,
    build_preprocessor,
)
from src.models.trainer import ModelBenchmarkTrainer

__all__ = [
    "build_preprocessor",
    "build_model_pipeline",
    "ALL_FEATURE_COLUMNS",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "ModelBenchmarkTrainer",
    "ModelEvaluationMetrics",
    "compute_evaluation_metrics",
    "plot_model_comparisons",
]
