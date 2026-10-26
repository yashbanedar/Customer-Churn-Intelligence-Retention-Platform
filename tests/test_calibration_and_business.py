"""
Unit tests for Phase 5:
Probability calibration, Cost-benefit threshold optimization,
SHAP explainability, and causal uncertainty documentation.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.business.cost_matrix import CostBenefitMatrix
from src.business.threshold_optimizer import CostSensitiveThresholdOptimizer
from src.config import BASE_DIR, get_settings
from src.data.cleaner import clean_dataset
from src.data.ingest import load_raw_data
from src.models.calibration import (
    CalibrationAuditResult,
    audit_calibration_performance,
    compute_expected_calibration_error,
    fit_calibrated_pipeline,
    plot_calibration_reliability_curve,
)
from src.models.explainability import SHAPExplainerEngine, clean_feature_names
from src.models.pipeline import ALL_FEATURE_COLUMNS, build_model_pipeline
from src.models.trainer import ModelBenchmarkTrainer


@pytest.fixture
def clean_sample_data() -> pd.DataFrame:
    raw_df = load_raw_data()
    return clean_dataset(raw_df)


def test_cost_benefit_matrix_calculations():
    """
    Verifies financial payouts for confusion matrix quadrants and theoretical Bayes cutoff.
    """
    cm = CostBenefitMatrix(offer_cost=35.0, save_rate=0.22, customer_clv=450.0)
    # V(TP) = (0.22 * 450) - 35 = 99 - 35 = 64
    assert pytest.approx(cm.value_true_positive, 0.01) == 64.0
    # V(FP) = -35
    assert cm.value_false_positive == -35.0
    # Bayes theoretical cutoff = 35 / (0.22 * 450) = 35 / 99 ≈ 0.3535
    assert pytest.approx(cm.theoretical_optimal_threshold, 0.005) == 0.354

    # Calculate net savings from CM: 10 TP, 5 FP => (10 * 64) - (5 * 35) = 640 - 175 = 465
    net_saved = cm.calculate_net_savings_from_cm(true_positives=10, false_positives=5)
    assert pytest.approx(net_saved, 0.01) == 465.0


def test_expected_calibration_error_bounds():
    """
    Verifies that compute_expected_calibration_error returns a valid metric in [0, 1].
    """
    y_true = np.array([0, 0, 0, 1, 1, 1])
    # Perfect calibration
    y_prob_good = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
    ece_good = compute_expected_calibration_error(y_true, y_prob_good, n_bins=5)
    assert 0.0 <= ece_good <= 0.25

    # Poor calibration
    y_prob_bad = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
    ece_bad = compute_expected_calibration_error(y_true, y_prob_bad, n_bins=5)
    assert ece_bad > ece_good


def test_fit_calibrated_pipeline(clean_sample_data):
    """
    Verifies fitting CalibratedClassifierCV strictly on training slice.
    """
    X = clean_sample_data[ALL_FEATURE_COLUMNS].head(200)
    y = clean_sample_data["churn_label"].head(200)

    base_pipe = build_model_pipeline(model_type="logistic_regression", random_state=42)
    calibrated_model = fit_calibrated_pipeline(base_pipe, X, y, method="sigmoid", cv=2)

    assert isinstance(calibrated_model, CalibratedClassifierCV)
    probas = calibrated_model.predict_proba(X.tail(10))
    assert probas.shape == (10, 2)
    assert (probas >= 0.0).all() and (probas <= 1.0).all()


def test_threshold_optimizer_on_training_data():
    """
    Verifies threshold optimizer finds optimal threshold and evaluates on test set.
    """
    cm = CostBenefitMatrix(offer_cost=35.0, save_rate=0.22, customer_clv=450.0)
    optimizer = CostSensitiveThresholdOptimizer(cost_matrix=cm)

    y_train_true = np.array([0] * 70 + [1] * 30)
    # Probabilities correlate with label
    y_train_probs = np.concatenate([np.random.uniform(0.0, 0.4, 70), np.random.uniform(0.5, 0.9, 30)])

    opt_tau = optimizer.fit_optimal_threshold(y_train_true, y_train_probs)
    assert 0.05 <= opt_tau <= 0.95

    # Compare thresholds on test set
    y_test_true = np.array([0] * 35 + [1] * 15)
    y_test_probs = np.concatenate([np.random.uniform(0.0, 0.4, 35), np.random.uniform(0.5, 0.9, 15)])
    comparison = optimizer.compare_thresholds_on_test(y_test_true, y_test_probs)

    assert "naive_threshold_0_50" in comparison
    assert "cost_optimal_threshold" in comparison
    assert "net_savings_delta_usd" in comparison


def test_shap_explainer_engine(clean_sample_data):
    """
    Verifies SHAP TreeExplainer initialization and calculation on XGBoost pipeline.
    """
    X = clean_sample_data[ALL_FEATURE_COLUMNS]
    y = clean_sample_data["churn_label"]

    pipe = build_model_pipeline(model_type="xgboost", random_state=42)
    pipe.fit(X.head(150), y.head(150))

    engine = SHAPExplainerEngine(pipeline=pipe, X_train=X.head(150))
    assert len(engine.clean_feature_names) > 0

    explanation = engine.explain_dataset(X.tail(10))
    assert explanation.shape[0] == 10
    assert explanation.shape[1] == len(engine.clean_feature_names)


def test_causal_uncertainty_documentation_exists():
    """
    Verifies that docs/PROBABILITY_AND_CAUSAL_UNCERTAINTY.md exists and explains key principles.
    """
    doc_path = BASE_DIR / "docs" / "PROBABILITY_AND_CAUSAL_UNCERTAINTY.md"
    assert doc_path.exists(), f"Document missing: {doc_path}"

    text = doc_path.read_text(encoding="utf-8")
    assert "Probability \u2260 Predestination" in text or "Probability != Predestination" in text or "Aleatoric Uncertainty" in text
    assert "Epistemic Uncertainty" in text
    assert "Intervention Paradox" in text
    assert "Punitive" in text
