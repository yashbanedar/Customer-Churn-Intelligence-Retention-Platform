"""
CLI Runner and Pipeline Orchestrator for Phase 4 ML Model Benchmarking.
Executes stratified data splitting, 5-fold cross-validation, test set evaluation,
serialization of fitted pipelines, and generation of comparative reports.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Union

from src.config import BASE_DIR, get_settings
from src.models.trainer import ModelBenchmarkTrainer


logger = logging.getLogger("src.models.train_models")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_training_pipeline(
    n_splits: int = 5,
    test_size: float = 0.20,
    random_state: int = 42,
    output_models_dir: Optional[Union[str, Path]] = None,
    output_metrics_dir: Optional[Union[str, Path]] = None,
    output_figures_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Executes complete Phase 4 Machine Learning training pipeline:
    1. Loads pre-validated data and checks class imbalance.
    2. Performs stratified train/test split (test set kept untouched).
    3. Runs 5-fold Stratified Cross-Validation on training data across Logistic Regression, Random Forest, and XGBoost.
    4. Evaluates all models on held-out test set for PR-AUC, ROC-AUC, F1, Precision, Recall, Confusion Matrix.
    5. Saves fitted pipelines (including best model) to artifacts/models/.
    6. Renders ROC curves, PR curves, and confusion matrix charts to artifacts/figures/.
    7. Saves comprehensive benchmark report to artifacts/metrics/.

    Returns:
        True if training succeeded, False otherwise.
    """
    logger.info("=" * 70)
    logger.info("STARTING PHASE 4: LEAKAGE-SAFE MACHINE LEARNING TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Initialize trainer with stratified split
    trainer = ModelBenchmarkTrainer(test_size=test_size, random_state=random_state)

    # 2. Run 5-fold Stratified Cross-Validation on training data
    logger.info("--> STEP 1: Running 5-Fold Stratified Cross-Validation on Training Folds...")
    cv_results = trainer.run_cross_validation(n_splits=n_splits)

    # 3. Fit models on training data and evaluate on held-out test set
    logger.info("--> STEP 2: Fitting Full Pipelines & Evaluating on Untouched Test Set...")
    fitted_pipelines, test_metrics, raw_predictions = trainer.train_and_evaluate_test_set()

    # 4. Serialize artifacts and compile benchmark report
    logger.info("--> STEP 3: Serializing Models, Comparison Charts & Benchmark Reports...")
    benchmark_data = trainer.save_benchmark_artifacts(
        fitted_pipelines=fitted_pipelines,
        cv_results=cv_results,
        test_metrics=test_metrics,
        raw_predictions=raw_predictions,
        output_models_dir=output_models_dir,
        output_metrics_dir=output_metrics_dir,
        output_figures_dir=output_figures_dir,
    )

    best_m = benchmark_data["metadata"]["best_model"]
    best_pr = benchmark_data["test_set_measured_metrics"][best_m]["pr_auc"]
    best_roc = benchmark_data["test_set_measured_metrics"][best_m]["roc_auc"]

    logger.info("=" * 70)
    logger.info(f"PHASE 4 ML BENCHMARKING COMPLETED SUCCESSFULLY!")
    logger.info(f"Selected Production Model: {best_m.upper()} | Test PR-AUC = {best_pr:.4f} | Test ROC-AUC = {best_roc:.4f}")
    logger.info("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark Telco Churn ML pipelines.")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of cross-validation splits (default: 5).")
    parser.add_argument("--test-size", type=float, default=0.20, help="Fraction for held-out test set (default: 0.20).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--models-dir", type=str, help="Custom directory to save serialized models.")
    parser.add_argument("--metrics-dir", type=str, help="Custom directory to save benchmark reports.")
    parser.add_argument("--figures-dir", type=str, help="Custom directory to save evaluation figures.")

    args = parser.parse_args()

    success = run_training_pipeline(
        n_splits=args.cv_splits,
        test_size=args.test_size,
        random_state=args.seed,
        output_models_dir=args.models_dir,
        output_metrics_dir=args.metrics_dir,
        output_figures_dir=args.figures_dir,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
