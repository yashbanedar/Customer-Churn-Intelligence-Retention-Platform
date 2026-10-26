"""
Cost-Sensitive Decision Threshold Optimizer.
Sweeps classification decision thresholds to maximize Net Saved Revenue
under an explicit financial cost matrix.
Tuned strictly on training cross-validation folds; evaluated on untouched test partition.
"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.business.cost_matrix import CostBenefitMatrix
from src.config import BASE_DIR, get_settings


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class ThresholdEvaluationPoint:
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    interventions_count: int
    total_campaign_cost: float
    gross_saved_revenue: float
    net_saved_revenue: float
    roi_pct: float


class CostSensitiveThresholdOptimizer:
    """
    Optimizes decision threshold based on financial return rather than naive 0.50 accuracy.
    """

    def __init__(self, cost_matrix: Optional[CostBenefitMatrix] = None):
        self.cost_matrix = cost_matrix or CostBenefitMatrix()
        self.optimal_threshold: float = 0.50
        self.training_curve_: List[ThresholdEvaluationPoint] = []

    def fit_optimal_threshold(
        self,
        y_train_true: np.ndarray,
        y_train_cv_probs: np.ndarray,
        threshold_range: Tuple[float, float] = (0.05, 0.95),
        steps: int = 91,
    ) -> float:
        """
        Sweeps candidate decision thresholds strictly across training out-of-fold predictions.
        Selects threshold maximizing Net Saved Revenue.

        Args:
            y_train_true: Training ground truth labels.
            y_train_cv_probs: Out-of-fold predicted probabilities from cross-validation.
            threshold_range: Min and max threshold bounds.
            steps: Number of grid steps.

        Returns:
            Optimal threshold value (float).
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], steps)
        best_net_savings = -float("inf")
        best_threshold = 0.50
        points = []

        for tau in thresholds:
            tau = round(float(tau), 3)
            y_pred = (y_train_cv_probs >= tau).astype(int)
            cm = confusion_matrix(y_train_true, y_pred)
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

            interventions = tp + fp
            campaign_cost = interventions * self.cost_matrix.offer_cost
            gross_saved = tp * self.cost_matrix.save_rate * self.cost_matrix.customer_clv
            net_saved = gross_saved - campaign_cost
            roi = (net_saved / campaign_cost * 100.0) if campaign_cost > 0 else 0.0

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            pt = ThresholdEvaluationPoint(
                threshold=tau,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1, 4),
                interventions_count=interventions,
                total_campaign_cost=round(campaign_cost, 2),
                gross_saved_revenue=round(gross_saved, 2),
                net_saved_revenue=round(net_saved, 2),
                roi_pct=round(roi, 2),
            )
            points.append(pt)

            if net_saved > best_net_savings:
                best_net_savings = net_saved
                best_threshold = tau

        self.optimal_threshold = best_threshold
        self.training_curve_ = points
        logger.info(
            f"Threshold Tuning on Training Folds Complete: "
            f"Optimal Threshold = {self.optimal_threshold:.3f} (Max CV Net Savings = ${best_net_savings:,.2f})"
        )
        return self.optimal_threshold

    def evaluate_threshold_on_test(
        self,
        y_test_true: np.ndarray,
        y_test_probs: np.ndarray,
        threshold: float,
    ) -> ThresholdEvaluationPoint:
        """
        Evaluates a specified threshold on the held-out test partition.
        """
        y_pred = (y_test_probs >= threshold).astype(int)
        cm = confusion_matrix(y_test_true, y_pred)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        interventions = tp + fp
        campaign_cost = interventions * self.cost_matrix.offer_cost
        gross_saved = tp * self.cost_matrix.save_rate * self.cost_matrix.customer_clv
        net_saved = gross_saved - campaign_cost
        roi = (net_saved / campaign_cost * 100.0) if campaign_cost > 0 else 0.0

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        return ThresholdEvaluationPoint(
            threshold=round(float(threshold), 3),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            interventions_count=interventions,
            total_campaign_cost=round(campaign_cost, 2),
            gross_saved_revenue=round(gross_saved, 2),
            net_saved_revenue=round(net_saved, 2),
            roi_pct=round(roi, 2),
        )

    def compare_thresholds_on_test(
        self,
        y_test_true: np.ndarray,
        y_test_probs: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compares naive threshold (0.50) vs cost-optimal threshold on the held-out test set.
        """
        naive_point = self.evaluate_threshold_on_test(y_test_true, y_test_probs, threshold=0.50)
        optimal_point = self.evaluate_threshold_on_test(y_test_true, y_test_probs, threshold=self.optimal_threshold)

        delta_savings = optimal_point.net_saved_revenue - naive_point.net_saved_revenue
        pct_lift = (delta_savings / abs(naive_point.net_saved_revenue)) * 100.0 if naive_point.net_saved_revenue != 0 else 0.0

        comparison = {
            "naive_threshold_0_50": asdict(naive_point),
            "cost_optimal_threshold": asdict(optimal_point),
            "net_savings_delta_usd": round(delta_savings, 2),
            "net_savings_lift_pct": round(pct_lift, 2),
            "additional_churners_saved": round(
                (optimal_point.true_positives - naive_point.true_positives) * self.cost_matrix.save_rate, 1
            ),
        }
        logger.info(
            f"Test Set Threshold Comparison: Naive (0.50) Net Savings = ${naive_point.net_saved_revenue:,.2f} vs "
            f"Optimal ({self.optimal_threshold:.3f}) Net Savings = ${optimal_point.net_saved_revenue:,.2f} "
            f"(Lift: +${delta_savings:,.2f} / {pct_lift:+.1f}%)"
        )
        return comparison

    def plot_threshold_curve(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Renders threshold optimization curve plotting Net Saved Revenue, Campaign Cost, and F1 across thresholds.
        """
        settings = get_settings()
        out_dir = Path(output_dir) if output_dir else BASE_DIR / "artifacts" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)

        taus = [pt.threshold for pt in self.training_curve_]
        net_savings = [pt.net_saved_revenue for pt in self.training_curve_]
        costs = [pt.total_campaign_cost for pt in self.training_curve_]
        f1s = [pt.f1 for pt in self.training_curve_]

        fig, ax1 = plt.subplots(figsize=(10, 5.5))

        # Primary Axis: Net Savings & Cost
        line1 = ax1.plot(taus, net_savings, color="#27ae60", linewidth=3, label="Net Saved Revenue ($)")
        line2 = ax1.plot(taus, costs, color="#e67e22", linewidth=2, linestyle="--", label="Campaign Outreach Cost ($)")
        ax1.set_xlabel("Decision Threshold (tau)", fontweight="bold")
        ax1.set_ylabel("Financial Dollar Impact ($)", color="#27ae60", fontweight="bold")
        ax1.tick_params(axis="y", labelcolor="#27ae60")

        # Mark optimal threshold
        opt_tau = self.optimal_threshold
        opt_idx = taus.index(opt_tau) if opt_tau in taus else int(np.argmax(net_savings))
        max_savings = net_savings[opt_idx]
        ax1.axvline(opt_tau, color="#d9534f", linestyle=":", linewidth=2, label=f"Optimal Threshold (tau* = {opt_tau:.3f})")
        ax1.scatter([opt_tau], [max_savings], color="#d9534f", s=100, zorder=5)
        ax1.annotate(
            f"Max Net Savings: \\${max_savings:,.0f}\nat tau* = {opt_tau:.3f}",
            xy=(opt_tau, max_savings),
            xytext=(15, 10),
            textcoords="offset points",
            fontweight="bold",
            color="#2c3e50",
            arrowprops=dict(arrowstyle="->", color="#d9534f", lw=1.5),
        )

        # Mark naive threshold 0.50
        ax1.axvline(0.50, color="gray", linestyle="-.", alpha=0.7, label="Naive Threshold (0.50)")

        # Secondary Axis: F1 Score
        ax2 = ax1.twinx()
        line3 = ax2.plot(taus, f1s, color="#2980b9", linewidth=2, linestyle=":", label="F1 Score")
        ax2.set_ylabel("F1 Score", color="#2980b9", fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#2980b9")
        ax2.set_ylim([0, 1.0])

        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right", frameon=True)
        ax1.grid(True, linestyle=":", alpha=0.6)
        plt.title(
            f"Cost-Sensitive Threshold Optimization: Maximizing Net Retained Value\n"
            f"(Offer Cost = \\${self.cost_matrix.offer_cost:.0f}, Save Rate = {self.cost_matrix.save_rate*100:.0f}%, CLV = \\${self.cost_matrix.customer_clv:.0f})",
            pad=12,
            fontweight="bold",
        )
        plt.tight_layout()

        out_path = out_dir / "12_threshold_net_savings_curve.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Rendered threshold net savings curve: {out_path}")
        return out_path
