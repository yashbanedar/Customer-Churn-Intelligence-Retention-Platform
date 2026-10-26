"""
Probability Calibration Evaluation and Model Wrapping.
Provides reliability curve calculation, Brier score comparison before/after calibration,
and CalibratedClassifierCV encapsulation.
Keeps final test partition strictly out of calibration tuning.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from src.config import BASE_DIR, get_settings


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class CalibrationAuditResult:
    model_name: str
    uncalibrated_brier: float
    calibrated_brier: float
    brier_improvement_pct: float
    uncalibrated_log_loss: float
    calibrated_log_loss: float
    log_loss_improvement_pct: float
    roc_auc_test: float
    expected_calibration_error: float


def fit_calibrated_pipeline(
    base_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Fits a calibrated classifier using strictly training data via internal cross-validation folds.
    The held-out test set remains completely unobserved.

    Args:
        base_pipeline: Uncalibrated scikit-learn Pipeline (preprocessing + classifier).
        X_train: Training features.
        y_train: Training binary labels.
        method: 'sigmoid' (Platt Scaling) or 'isotonic'.
        cv: Internal cross-validation folds on train set.

    Returns:
        Fitted CalibratedClassifierCV wrapper.
    """
    logger.info(f"Fitting CalibratedClassifierCV ({method}, cv={cv}) on {len(X_train):,} training records...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method=method,
        cv=cv,
        n_jobs=-1,
    )
    calibrated_model.fit(X_train, y_train)
    logger.info("Probability calibration successfully completed on training data.")
    return calibrated_model


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE):
    Weighted average difference between predicted bin confidence and empirical accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def audit_calibration_performance(
    uncalibrated_pipe: Pipeline,
    calibrated_model: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "XGBoost",
) -> Tuple[CalibrationAuditResult, Dict[str, np.ndarray]]:
    """
    Evaluates probability calibration on the held-out test set before vs after calibration.
    Computes Brier Score, Log Loss, and Expected Calibration Error (ECE).
    """
    y_t = y_test.values.astype(int)

    # 1. Uncalibrated predictions
    uncal_probs = uncalibrated_pipe.predict_proba(X_test)[:, 1]
    uncal_brier = float(brier_score_loss(y_t, uncal_probs))
    uncal_logloss = float(log_loss(y_t, uncal_probs))

    # 2. Calibrated predictions
    cal_probs = calibrated_model.predict_proba(X_test)[:, 1]
    cal_brier = float(brier_score_loss(y_t, cal_probs))
    cal_logloss = float(log_loss(y_t, cal_probs))
    roc_auc = float(roc_auc_score(y_t, cal_probs))

    brier_impr = ((uncal_brier - cal_brier) / uncal_brier) * 100.0
    logloss_impr = ((uncal_logloss - cal_logloss) / uncal_logloss) * 100.0
    ece = compute_expected_calibration_error(y_t, cal_probs, n_bins=10)

    result = CalibrationAuditResult(
        model_name=model_name,
        uncalibrated_brier=round(uncal_brier, 4),
        calibrated_brier=round(cal_brier, 4),
        brier_improvement_pct=round(brier_impr, 2),
        uncalibrated_log_loss=round(uncal_logloss, 4),
        calibrated_log_loss=round(cal_logloss, 4),
        log_loss_improvement_pct=round(logloss_impr, 2),
        roc_auc_test=round(roc_auc, 4),
        expected_calibration_error=round(ece, 4),
    )

    logger.info(
        f"[{model_name} Calibration Audit] Brier Score: {uncal_brier:.4f} -> {cal_brier:.4f} "
        f"({brier_impr:+.2f}% improvement). ECE={ece:.4f}"
    )

    probs_dict = {
        "y_true": y_t,
        "uncalibrated_probs": uncal_probs,
        "calibrated_probs": cal_probs,
    }
    return result, probs_dict


def plot_calibration_reliability_curve(
    y_true: np.ndarray,
    uncal_probs: np.ndarray,
    cal_probs: np.ndarray,
    model_name: str = "XGBoost",
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Renders publication-grade Reliability Diagram (Calibration Curve) and Probability Histograms
    comparing uncalibrated vs calibrated probabilities.
    """
    settings = get_settings()
    out_dir = Path(output_dir) if output_dir else BASE_DIR / "artifacts" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Reliability Curve (Calibration curve)
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, uncal_probs, n_bins=10, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, cal_probs, n_bins=10, strategy="uniform")

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration (Ideal 45°)")
    axes[0].plot(
        prob_pred_uncal,
        prob_true_uncal,
        marker="o",
        linewidth=2,
        color="#d9534f",
        label=f"Raw {model_name} (Brier={brier_score_loss(y_true, uncal_probs):.4f})",
    )
    axes[0].plot(
        prob_pred_cal,
        prob_true_cal,
        marker="s",
        linewidth=2.5,
        color="#27ae60",
        label=f"Calibrated {model_name} (Brier={brier_score_loss(y_true, cal_probs):.4f})",
    )

    axes[0].set_title(f"Reliability Diagram: {model_name} Before vs After Calibration", pad=12, fontweight="bold")
    axes[0].set_xlabel("Mean Predicted Probability (Confidence)", fontweight="bold")
    axes[0].set_ylabel("Empirical Fraction of Positives (Accuracy)", fontweight="bold")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.0])
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Right: Distribution of Predicted Probabilities
    axes[1].hist(
        uncal_probs,
        bins=25,
        range=(0, 1),
        alpha=0.45,
        color="#d9534f",
        label="Raw Uncalibrated Probabilities",
        edgecolor="black",
    )
    axes[1].hist(
        cal_probs,
        bins=25,
        range=(0, 1),
        alpha=0.55,
        color="#27ae60",
        label="Calibrated Probabilities (Platt Scaling)",
        edgecolor="black",
    )
    axes[1].set_title("Probability Score Distributions Across Test Accounts", pad=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Churn Probability", fontweight="bold")
    axes[1].set_ylabel("Customer Account Count", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Empirical Probability Calibration Analysis (Held-Out Test Set)", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = out_dir / "11_calibration_reliability_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Rendered calibration reliability curve: {out_path}")
    return out_path
