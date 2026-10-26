"""
SHAP (SHapley Additive exPlanations) Explainability Engine.
Generates global feature attributions (Beeswarm, Bar) and local customer waterfall explanations.
Serializes TreeExplainer artifacts for downstream API inference.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import BASE_DIR, get_settings


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def clean_feature_names(raw_names: List[str]) -> List[str]:
    """
    Cleans ColumnTransformer prefixes ('num__', 'cat__') into clean business labels.
    """
    cleaned = []
    for name in raw_names:
        clean = name.replace("num__", "").replace("cat__", "")
        clean = clean.replace("_", " ").title()
        cleaned.append(clean)
    return cleaned


class SHAPExplainerEngine:
    """
    Manages SHAP tree explanation computation and visualization.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        settings = get_settings()
        self.output_dir = Path(output_dir) if output_dir else BASE_DIR / "artifacts" / "explainers"
        self.figures_dir = BASE_DIR / "artifacts" / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Transform background training sample for tree explainer
        logger.info("Transforming feature matrix for SHAP TreeExplainer...")
        self.raw_feature_names = list(self.preprocessor.get_feature_names_out())
        self.clean_feature_names = clean_feature_names(self.raw_feature_names)

        # TreeExplainer on XGBoost estimator
        self.explainer = shap.TreeExplainer(self.classifier)
        logger.info("SHAP TreeExplainer successfully initialized.")

    def explain_dataset(
        self,
        X_eval: pd.DataFrame,
    ) -> shap.Explanation:
        """
        Computes SHAP Explanation object for an evaluation dataset.
        """
        X_trans = self.preprocessor.transform(X_eval)
        # Create an Explanation object with clean feature names
        shap_values = self.explainer(X_trans)
        shap_values.feature_names = self.clean_feature_names
        return shap_values

    def plot_global_explanations(
        self,
        explanation: shap.Explanation,
        max_display: int = 15,
    ) -> Dict[str, Path]:
        """
        Renders Global SHAP Summary Plots (Beeswarm and Bar Chart).
        """
        # 1. Beeswarm Plot
        fig_bee = plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)
        plt.title(f"SHAP Global Feature Attributions (Top {max_display} Impactful Drivers)", pad=15, fontweight="bold")
        plt.tight_layout()
        bee_path = self.figures_dir / "13_shap_global_beeswarm.png"
        plt.savefig(bee_path, dpi=200, bbox_inches="tight")
        plt.close(fig_bee)

        # 2. Bar Chart of Mean Absolute SHAP values
        fig_bar = plt.figure(figsize=(10, 6))
        shap.plots.bar(explanation, max_display=max_display, show=False)
        plt.title(f"Mean |SHAP Value| (Average Marginal Contribution to Churn)", pad=15, fontweight="bold")
        plt.tight_layout()
        bar_path = self.figures_dir / "14_shap_global_bar.png"
        plt.savefig(bar_path, dpi=200, bbox_inches="tight")
        plt.close(fig_bar)

        logger.info(f"Rendered global SHAP plots: {bee_path.name}, {bar_path.name}")
        return {
            "shap_beeswarm": bee_path,
            "shap_bar": bar_path,
        }

    def plot_local_waterfall_explanations(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        explanation: shap.Explanation,
    ) -> Dict[str, Path]:
        """
        Renders local waterfall explanations for representative individual customer personas:
        - High-Risk True Churner
        - Low-Risk Loyal Customer
        - Borderline Uncertain Customer
        """
        # Calculate predicted probabilities on test
        probs = self.pipeline.predict_proba(X_test)[:, 1]
        y_t = y_test.values

        # 1. Identify High-Risk True Churner (high prob, actual=1)
        high_risk_candidates = np.where((probs > 0.75) & (y_t == 1))[0]
        high_risk_idx = int(high_risk_candidates[0]) if len(high_risk_candidates) > 0 else int(np.argmax(probs))

        # 2. Identify Low-Risk Loyal Customer (low prob, actual=0)
        low_risk_candidates = np.where((probs < 0.10) & (y_t == 0))[0]
        low_risk_idx = int(low_risk_candidates[0]) if len(low_risk_candidates) > 0 else int(np.argmin(probs))

        # 3. Identify Borderline Customer (prob close to 0.40 - 0.50)
        border_candidates = np.argsort(np.abs(probs - 0.45))
        border_idx = int(border_candidates[0])

        figures = {}

        # Render High-Risk Waterfall
        fig_high = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[high_risk_idx], max_display=10, show=False)
        plt.title(f"Customer Case A: High-Risk Flight Alarm (Predicted Churn Prob = {probs[high_risk_idx]:.1%})", pad=12, fontweight="bold")
        plt.tight_layout()
        high_path = self.figures_dir / "15_shap_waterfall_high_risk.png"
        plt.savefig(high_path, dpi=200, bbox_inches="tight")
        plt.close(fig_high)
        figures["waterfall_high_risk"] = high_path

        # Render Low-Risk Waterfall
        fig_low = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[low_risk_idx], max_display=10, show=False)
        plt.title(f"Customer Case B: Low-Risk Anchored Account (Predicted Churn Prob = {probs[low_risk_idx]:.1%})", pad=12, fontweight="bold")
        plt.tight_layout()
        low_path = self.figures_dir / "16_shap_waterfall_low_risk.png"
        plt.savefig(low_path, dpi=200, bbox_inches="tight")
        plt.close(fig_low)
        figures["waterfall_low_risk"] = low_path

        # Render Borderline Waterfall
        fig_border = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[border_idx], max_display=10, show=False)
        plt.title(f"Customer Case C: Borderline Decision Boundary (Predicted Churn Prob = {probs[border_idx]:.1%})", pad=12, fontweight="bold")
        plt.tight_layout()
        border_path = self.figures_dir / "17_shap_waterfall_borderline.png"
        plt.savefig(border_path, dpi=200, bbox_inches="tight")
        plt.close(fig_border)
        figures["waterfall_borderline"] = border_path

        logger.info(f"Rendered local SHAP waterfalls: {high_path.name}, {low_path.name}, {border_path.name}")
        return figures

    def save_explainer_artifacts(self) -> Dict[str, Path]:
        """
        Serializes the TreeExplainer and feature names to artifacts/explainers/.
        """
        explainer_path = self.output_dir / "shap_tree_explainer.joblib"
        joblib.dump(self.explainer, explainer_path)

        names_path = self.output_dir / "feature_names.json"
        with open(names_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "raw_feature_names": self.raw_feature_names,
                    "clean_feature_names": self.clean_feature_names,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved SHAP explainer artifacts to: {self.output_dir}")
        return {
            "explainer": explainer_path,
            "feature_names": names_path,
        }
