"""
Model Performance Evaluation Engine.
Calculates PR-AUC, ROC-AUC, Precision, Recall, F1, Brier Score, and Confusion Matrix metrics.
Renders comparative evaluation figures (ROC curves, PR curves, Confusion Matrices).
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import BASE_DIR, get_settings


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class ModelEvaluationMetrics:
    model_name: str
    pr_auc: float
    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    brier_score: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    specificity: float


def compute_evaluation_metrics(
    y_true: Union[np.ndarray, list],
    y_proba: Union[np.ndarray, list],
    threshold: float = 0.5,
    model_name: str = "Model",
) -> ModelEvaluationMetrics:
    """
    Computes rigorous measured classification metrics on probability predictions.

    Args:
        y_true: Ground truth binary target (0 or 1).
        y_proba: Predicted positive class probabilities.
        threshold: Decision threshold for discrete classification.
        model_name: Identifier for model.

    Returns:
        Structured ModelEvaluationMetrics dataclass instance.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_proba).astype(float)
    y_pred = (y_p >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_t, y_p))
    roc_auc = float(roc_auc_score(y_t, y_p))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    acc = float(accuracy_score(y_t, y_pred))
    brier = float(brier_score_loss(y_t, y_p))

    cm = confusion_matrix(y_t, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return ModelEvaluationMetrics(
        model_name=model_name,
        pr_auc=round(pr_auc, 4),
        roc_auc=round(roc_auc, 4),
        f1=round(f1, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        accuracy=round(acc, 4),
        brier_score=round(brier, 4),
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        specificity=round(spec, 4),
    )


def plot_model_comparisons(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """
    Generates comparative ROC Curves, Precision-Recall Curves, and Confusion Matrices.

    Args:
        results: Dict mapping model_name -> (y_true, y_proba).
        output_dir: Folder to save figures. Defaults to artifacts/figures/.

    Returns:
        Dict mapping chart name to saved file Path.
    """
    settings = get_settings()
    out_dir = Path(output_dir) if output_dir else BASE_DIR / "artifacts" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "Logistic Regression": "#2980b9",
        "Random Forest": "#27ae60",
        "XGBoost": "#e74c3c",
    }
    fallback_palette = ["#2980b9", "#27ae60", "#e74c3c", "#8e44ad"]

    # 1. ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for idx, (name, (y_t, y_p)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_t, y_p)
        auc_score = roc_auc_score(y_t, y_p)
        c = colors.get(name, fallback_palette[idx % len(fallback_palette)])
        ax_roc.plot(fpr, tpr, label=f"{name} (ROC-AUC = {auc_score:.3f})", color=c, linewidth=2.5)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier (0.500)")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Comparison", pad=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontweight="bold")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.legend(loc="lower right", frameon=True)
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    roc_path = out_dir / "08_model_roc_curves.png"
    fig_roc.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close(fig_roc)

    # 2. Precision-Recall Curves
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    baseline_churn = np.mean(list(results.values())[0][0])
    for idx, (name, (y_t, y_p)) in enumerate(results.items()):
        prec, rec, _ = precision_recall_curve(y_t, y_p)
        pr_score = average_precision_score(y_t, y_p)
        c = colors.get(name, fallback_palette[idx % len(fallback_palette)])
        ax_pr.plot(rec, prec, label=f"{name} (PR-AUC = {pr_score:.3f})", color=c, linewidth=2.5)

    ax_pr.axhline(baseline_churn, linestyle="--", color="gray", label=f"No-Skill Baseline ({baseline_churn:.3f})")
    ax_pr.set_title("Precision-Recall (PR) Curve Comparison (Imbalanced Focus)", pad=12, fontweight="bold")
    ax_pr.set_xlabel("Recall (Coverage of Actual Churners)", fontweight="bold")
    ax_pr.set_ylabel("Precision (Accuracy of Churn Alarms)", fontweight="bold")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.legend(loc="upper right", frameon=True)
    ax_pr.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    pr_path = out_dir / "09_model_pr_curves.png"
    fig_pr.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close(fig_pr)

    # 3. Side-by-Side Confusion Matrices
    n_models = len(results)
    fig_cm, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for idx, (name, (y_t, y_p)) in enumerate(results.items()):
        y_pred = (y_p >= 0.5).astype(int)
        cm = confusion_matrix(y_t, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"],
            ax=axes[idx],
        )
        axes[idx].set_title(f"{name}\nThreshold = 0.50", fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontweight="bold")
        if idx == 0:
            axes[idx].set_ylabel("Actual Ground Truth", fontweight="bold")

    plt.suptitle("Test Set Confusion Matrices Across Evaluated Architectures", fontsize=13, y=1.03)
    plt.tight_layout()
    cm_path = out_dir / "10_confusion_matrices.png"
    fig_cm.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig_cm)

    logger.info(f"Rendered model comparison plots to: {out_dir}")
    return {
        "roc_curves": roc_path,
        "pr_curves": pr_path,
        "confusion_matrices": cm_path,
    }
