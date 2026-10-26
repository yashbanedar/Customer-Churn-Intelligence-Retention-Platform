"""
Unit tests for Phase 4 Machine Learning Pipeline:
Tests preprocessor construction, pipeline factory, stratified splitting,
leakage isolation, evaluation metric accuracy, and pipeline serialization.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import BASE_DIR, get_settings
from src.data.cleaner import clean_dataset
from src.data.ingest import load_raw_data
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


@pytest.fixture
def clean_sample_data() -> pd.DataFrame:
    raw_df = load_raw_data()
    return clean_dataset(raw_df)


def test_build_preprocessor_structure():
    """
    Verifies that the ColumnTransformer handles both numeric and categorical features
    and drops unspecified remainder columns.
    """
    preprocessor = build_preprocessor()
    assert preprocessor.remainder == "drop"
    transformer_names = [t[0] for t in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_build_model_pipeline_factory():
    """
    Verifies construction of all supported classifier pipelines and error on unsupported types.
    """
    for m_type in ["logistic_regression", "random_forest", "xgboost"]:
        pipe = build_model_pipeline(model_type=m_type)
        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps

    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_model_pipeline(model_type="unsupported_model")


def test_trainer_stratified_split_and_zero_leakage(clean_sample_data):
    """
    Verifies that train/test split is strictly stratified, has zero index overlap,
    and isolates customer_id from feature space.
    """
    trainer = ModelBenchmarkTrainer(df=clean_sample_data, test_size=0.20, random_state=42)

    assert len(trainer.X_train) == 5634
    assert len(trainer.X_test) == 1409
    assert len(trainer.y_train) == 5634
    assert len(trainer.y_test) == 1409

    # Zero index intersection
    train_idx = set(trainer.X_train.index)
    test_idx = set(trainer.X_test.index)
    assert len(train_idx.intersection(test_idx)) == 0

    # customer_id exclusion
    assert "customer_id" not in trainer.X_train.columns
    assert "customer_id" not in trainer.X_test.columns

    # Class distribution match (stratification)
    assert abs(trainer.y_train.mean() - trainer.y_test.mean()) < 0.005


def test_compute_evaluation_metrics_identities():
    """
    Verifies mathematical identities of confusion matrix and classification metrics.
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Predicted probabilities
    y_proba = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.8, 0.4, 0.9])

    metrics = compute_evaluation_metrics(y_true, y_proba, threshold=0.5, model_name="TestModel")
    assert isinstance(metrics, ModelEvaluationMetrics)
    assert metrics.true_negatives + metrics.false_positives + metrics.false_negatives + metrics.true_positives == len(y_true)
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.brier_score <= 1.0


def test_pipeline_fit_and_inference(clean_sample_data):
    """
    Fits Logistic Regression pipeline on sample data and verifies predict_proba shape and range.
    """
    X = clean_sample_data[ALL_FEATURE_COLUMNS]
    y = clean_sample_data["churn_label"]

    pipeline = build_model_pipeline(model_type="logistic_regression", random_state=42)
    pipeline.fit(X.head(100), y.head(100))

    # Test prediction on hold-out slice
    sample_test = X.tail(10)
    probas = pipeline.predict_proba(sample_test)
    assert probas.shape == (10, 2)
    assert (probas >= 0.0).all() and (probas <= 1.0).all()
    # Sum of probabilities must equal 1.0
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(10), rtol=1e-5)


def test_serialized_best_pipeline_loads_and_predicts():
    """
    Verifies that the serialized best model pipeline exists in artifacts/models/
    and successfully generates predictions on sample inputs.
    """
    settings = get_settings()
    best_pipe_path = BASE_DIR / settings.model_artifacts_dir / "best_churn_pipeline.joblib"
    assert best_pipe_path.exists(), f"Serialized pipeline not found at {best_pipe_path}"

    loaded_pipe = joblib.load(best_pipe_path)
    assert isinstance(loaded_pipe, Pipeline)

    raw_df = load_raw_data()
    clean_df = clean_dataset(raw_df)
    sample_features = clean_df[ALL_FEATURE_COLUMNS].head(5)

    preds = loaded_pipe.predict_proba(sample_features)[:, 1]
    assert len(preds) == 5
    assert (preds >= 0.0).all() and (preds <= 1.0).all()


def test_plot_model_comparisons(tmp_path):
    """
    Verifies that plot_model_comparisons creates ROC, PR, and confusion matrix figures.
    """
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0])
    dummy_results = {
        "Logistic Regression": (y_true, np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.2, 0.1])),
        "XGBoost": (y_true, np.array([0.05, 0.95, 0.1, 0.85, 0.2, 0.75, 0.1, 0.05])),
    }
    figs = plot_model_comparisons(dummy_results, output_dir=tmp_path)
    assert "roc_curves" in figs and figs["roc_curves"].exists()
    assert "pr_curves" in figs and figs["pr_curves"].exists()
    assert "confusion_matrices" in figs and figs["confusion_matrices"].exists()


def test_run_training_pipeline_end_to_end(tmp_path):
    """
    Verifies end-to-end execution of run_training_pipeline CLI entrypoint with isolated output directories.
    """
    from src.models.train_models import run_training_pipeline

    m_dir = tmp_path / "test_models"
    met_dir = tmp_path / "test_metrics"
    fig_dir = tmp_path / "test_figures"

    success = run_training_pipeline(
        n_splits=2,
        test_size=0.20,
        random_state=42,
        output_models_dir=m_dir,
        output_metrics_dir=met_dir,
        output_figures_dir=fig_dir,
    )

    assert success is True
    assert (m_dir / "best_churn_pipeline.joblib").exists()
    assert (m_dir / "xgboost_pipeline.joblib").exists()
    assert (met_dir / "model_benchmark_results.json").exists()
    assert (met_dir / "model_benchmark_report.md").exists()
    assert (fig_dir / "08_model_roc_curves.png").exists()

