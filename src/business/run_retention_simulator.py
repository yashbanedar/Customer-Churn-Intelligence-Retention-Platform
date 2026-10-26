"""
Retention Budget Simulator Execution Script.
Runs the transparent retention simulator on the Telco customer dataset using the
calibrated probability pipeline, evaluates multi-scenario what-if analyses, sweeps budget
allocations, and outputs metrics, tables, and visualization charts.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.business.retention_simulator import (
    CAUSAL_DISCLAIMER,
    CLVCalculationMode,
    RankingStrategy,
    RetentionAssumptionConfig,
    RetentionBudgetSimulator,
)
from src.config import BASE_DIR, get_settings


logger = logging.getLogger("RetentionSimulatorRunner")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_simulator_pipeline(
    dataset_path: Path,
    model_path: Path,
    budget: float = 10000.0,
    cost_per_offer: float = 35.0,
    offer_success_rate: float = 0.22,
    fixed_clv: float = 450.0,
    clv_mode_str: str = "fixed",
    use_test_partition_only: bool = True,
    output_dir: Path = BASE_DIR / "artifacts",
) -> Dict[str, Any]:
    """
    Executes the end-to-end retention budget simulation workflow.
    """
    logger.info("=" * 80)
    logger.info("STARTING PHASE 6: TRANSPARENT RETENTION-BUDGET SIMULATOR")
    logger.info("=" * 80)

    # 1. Load Data
    logger.info(f"Loading customer dataset from: {dataset_path}")
    if dataset_path.suffix == ".parquet":
        df_all = pd.read_parquet(dataset_path)
    else:
        df_all = pd.read_csv(dataset_path)

    logger.info(f"Loaded {len(df_all):,} records with columns: {list(df_all.columns)}")

    # 2. Partition or select target cohort
    # Consistent 80/20 train/test split matching Phase 4 and Phase 5
    from src.models.trainer import ModelBenchmarkTrainer

    trainer = ModelBenchmarkTrainer(test_size=0.20, random_state=42)
    if use_test_partition_only:
        eval_df = trainer.df.loc[trainer.X_test.index].copy().reset_index(drop=True)
        cohort_description = f"Held-out Unobserved Test Partition (N={len(eval_df):,})"
        X_eval = trainer.X_test.copy().reset_index(drop=True)
    else:
        eval_df = trainer.df.copy().reset_index(drop=True)
        cohort_description = f"Full Active Customer Base (N={len(eval_df):,})"
        X_eval = trainer.X.copy().reset_index(drop=True)

    logger.info(f"Evaluation Cohort: {cohort_description}")

    # 3. Load Calibrated Model Pipeline
    logger.info(f"Loading calibrated model pipeline from: {model_path}")
    calibrated_pipeline = joblib.load(model_path)

    # Predict calibrated posterior probabilities
    logger.info("Computing calibrated posterior churn probabilities...")
    calibrated_probs = calibrated_pipeline.predict_proba(X_eval)[:, 1]
    logger.info(
        f"Calibrated Churn Probabilities: Mean={calibrated_probs.mean():.4f}, "
        f"Median={np.median(calibrated_probs):.4f}, Min={calibrated_probs.min():.4f}, Max={calibrated_probs.max():.4f}"
    )

    # 4. Initialize Baseline Assumption Configuration
    clv_mode = CLVCalculationMode(clv_mode_str.lower())
    baseline_cfg = RetentionAssumptionConfig(
        scenario_name="Baseline (Pragmatic)",
        budget=float(budget),
        cost_per_offer=float(cost_per_offer),
        assumed_offer_success_rate=float(offer_success_rate),
        clv_mode=clv_mode,
        fixed_clv=float(fixed_clv),
        clv_horizon_months=12,
        profit_margin=0.80,
        ranking_strategy=RankingStrategy.EXPECTED_NET_VALUE,
        require_positive_expected_value=True,
    )

    simulator = RetentionBudgetSimulator(config=baseline_cfg)

    # 5. Execute Baseline Simulation
    baseline_outcome, segmented_df = simulator.simulate(
        customers_df=eval_df,
        churn_probabilities=calibrated_probs,
        config=baseline_cfg,
    )

    # 6. Run Multi-Scenario Comparison
    scenarios: List[RetentionAssumptionConfig] = [
        RetentionAssumptionConfig(
            scenario_name="Conservative",
            budget=float(budget),
            cost_per_offer=45.0,
            assumed_offer_success_rate=0.12,
            clv_mode=CLVCalculationMode.FIXED,
            fixed_clv=400.0,
            ranking_strategy=RankingStrategy.EXPECTED_NET_VALUE,
            require_positive_expected_value=True,
        ),
        RetentionAssumptionConfig(
            scenario_name="Baseline (Pragmatic)",
            budget=float(budget),
            cost_per_offer=35.0,
            assumed_offer_success_rate=0.22,
            clv_mode=CLVCalculationMode.FIXED,
            fixed_clv=450.0,
            ranking_strategy=RankingStrategy.EXPECTED_NET_VALUE,
            require_positive_expected_value=True,
        ),
        RetentionAssumptionConfig(
            scenario_name="Optimistic",
            budget=float(budget),
            cost_per_offer=25.0,
            assumed_offer_success_rate=0.32,
            clv_mode=CLVCalculationMode.FIXED,
            fixed_clv=500.0,
            ranking_strategy=RankingStrategy.EXPECTED_NET_VALUE,
            require_positive_expected_value=True,
        ),
        RetentionAssumptionConfig(
            scenario_name="Customer-Specific CLV",
            budget=float(budget),
            cost_per_offer=35.0,
            assumed_offer_success_rate=0.22,
            clv_mode=CLVCalculationMode.CUSTOMER_SPECIFIC,
            clv_horizon_months=12,
            profit_margin=0.80,
            ranking_strategy=RankingStrategy.EXPECTED_NET_VALUE,
            require_positive_expected_value=True,
        ),
    ]

    logger.info("Running multi-scenario comparative evaluation...")
    comparison_df = simulator.compare_scenarios(
        customers_df=eval_df,
        churn_probabilities=calibrated_probs,
        scenarios=scenarios,
    )

    # 7. Run Budget Sensitivity Sweep
    sweep_budgets = [1000.0, 2500.0, 5000.0, 7500.0, 10000.0, 15000.0, 20000.0, 30000.0, 45000.0]
    logger.info(f"Sweeping budget sensitivity across levels: {sweep_budgets}...")
    budget_sweep_df = simulator.sweep_budget_sensitivity(
        customers_df=eval_df,
        churn_probabilities=calibrated_probs,
        budgets=sweep_budgets,
        base_config=baseline_cfg,
    )

    # 8. Render Visualizations
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    scenario_plot_path = fig_dir / "18_retention_budget_scenarios.png"
    simulator.plot_scenario_analysis(
        comparison_df=comparison_df,
        budget_sweep_df=budget_sweep_df,
        target_records_df=segmented_df,
        output_path=scenario_plot_path,
    )

    # 9. Save Transparent Segmented Target CSV
    processed_dir = BASE_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    targets_csv_path = processed_dir / "simulated_retention_targets.csv"
    segmented_df.to_csv(targets_csv_path, index=False)
    logger.info(f"Saved segmented customer targeting table to: {targets_csv_path}")

    # 10. Compile Reports & Summaries
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary_dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_description": cohort_description,
        "causal_disclaimer": CAUSAL_DISCLAIMER,
        "baseline_outcome": baseline_outcome.to_dict(),
        "scenarios_comparison": comparison_df.to_dict(orient="records"),
        "budget_sweep_sensitivity": budget_sweep_df.to_dict(orient="records"),
    }

    summary_json_path = metrics_dir / "retention_simulation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Saved simulation summary JSON to: {summary_json_path}")

    # Compile Markdown Audit Report
    report_md_path = metrics_dir / "retention_simulation_report.md"
    _generate_markdown_report(
        report_path=report_md_path,
        baseline_outcome=baseline_outcome,
        comparison_df=comparison_df,
        budget_sweep_df=budget_sweep_df,
        segmented_df=segmented_df,
        cohort_description=cohort_description,
    )
    logger.info(f"Saved simulation markdown report to: {report_md_path}")

    logger.info("=" * 80)
    logger.info("PHASE 6 SIMULATION PIPELINE COMPLETE")
    logger.info("=" * 80)

    return summary_dict


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Formats DataFrame as GitHub Flavored Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        row_str = [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines)


def _generate_markdown_report(
    report_path: Path,
    baseline_outcome: Any,
    comparison_df: pd.DataFrame,
    budget_sweep_df: pd.DataFrame,
    segmented_df: pd.DataFrame,
    cohort_description: str,
) -> None:
    """
    Generates a comprehensive markdown report detailing simulation findings,
    financial projections, transparent data lineage, and causal disclaimers.
    """
    targeted_subset = segmented_df[segmented_df["simulated_is_targeted"]]
    sample_top_targeted = targeted_subset.head(5)

    md_lines = [
        "# Phase 6: Transparent Retention Budget Simulation & Financial Optimization Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Cohort Evaluated**: `{cohort_description}`  ",
        "**Methodological Framework**: Expected Value Optimization under Explicit Hypothetical Assumptions  ",
        "",
        "---",
        "",
        "## ⚠️ Methodological Causal Uplift Disclaimer",
        "",
        "> [!IMPORTANT]",
        f"> **{CAUSAL_DISCLAIMER}**",
        "",
        "### Why We Cannot Claim Empirical Causal Uplift from Observational Data Alone:",
        "1. **Absence of Treatment Variation**: The historical IBM Telco dataset contains zero records of marketing interventions, proactive discounts, or retention outreach.",
        "2. **Confounding & The Sleeping Dog Effect**: Contacting a customer who had no intention of churning may actively trigger churn by reminding them of monthly bills or service dissatisfaction.",
        "3. **Observational Association vs. Counterfactual Response**: A customer with a $0.80$ predicted churn probability will not necessarily respond positively to a $35$ discount. Uplift modeling requires estimating the Conditional Average Treatment Effect (CATE) $\\tau(X) = \\mathbb{E}[Y(1) - Y(0) \\mid X]$, which strictly requires randomized trial data (A/B testing).",
        "",
        "---",
        "",
        "## 🔍 Transparent Data Segmentation & Lineage",
        "",
        "To prevent confusion between ground-truth business facts and hypothetical scenario modeling, every column in the targeting schema is strictly classified into one of four categories:",
        "",
        "| Category | Column Prefix | Definition & Source | Examples |",
        "| :--- | :--- | :--- | :--- |",
        "| **Observed Data** | `observed_*` | Ground-truth empirical facts from customer records | `customerID`, `MonthlyCharges`, `tenure`, `Contract` |",
        "| **Model Prediction** | `predicted_*` | Statistical inference output from calibrated ML model | `calibrated_churn_prob`, `risk_tier` |",
        "| **Hypothetical Assumption** | `assumed_*` | User-defined scenario inputs for planning | `offer_cost`, `assumed_success_rate`, `customer_clv` |",
        "| **Simulated Outcome** | `simulated_*` | Computed counterfactual financial projections | `expected_gross_saved`, `expected_net_value`, `is_targeted` |",
        "",
        "### Sample Top-Priority Interventions Table",
        "",
    ]

    # Render top 5 sample table
    display_cols = [
        "observed_customer_id",
        "observed_monthly_charges",
        "observed_contract_type",
        "predicted_calibrated_churn_prob",
        "assumed_customer_clv",
        "simulated_expected_gross_saved",
        "simulated_expected_net_value",
        "simulated_expected_roi_pct",
        "simulated_is_targeted",
    ]
    sample_df = sample_top_targeted[display_cols]
    md_lines.append(_df_to_markdown(sample_df))
    md_lines.append("")

    # Section 3: Baseline Financial Outcomes
    md_lines.extend([
        "---",
        "",
        "## 💰 Baseline Scenario Performance Outcomes",
        "",
        f"- **Campaign Budget Allocated**: `${baseline_outcome.budget_allocated:,.2f}`",
        f"- **Actual Campaign Spend**: `${baseline_outcome.budget_spent:,.2f}` ({baseline_outcome.budget_utilization_pct:.1f}% utilization)",
        f"- **Budget Remaining**: `${baseline_outcome.budget_remaining:,.2f}`",
        f"- **Candidate Pool Size**: `{baseline_outcome.total_candidate_pool:,}` accounts",
        f"- **Customers Targeted for Proactive Retention**: `{baseline_outcome.customers_targeted:,}` accounts ({baseline_outcome.targeting_rate_pct:.1f}% of pool)",
        f"- **Expected Accounts Saved (Counterfactual)**: `{baseline_outcome.expected_customers_saved:.1f}` accounts",
        f"- **Cost per Saved Account**: `${baseline_outcome.cost_per_saved_customer:,.2f}`",
        f"- **Expected Gross Revenue Saved**: `${baseline_outcome.expected_gross_revenue_saved:,.2f}`",
        f"- **Expected Net Saved Revenue (Profit)**: `${baseline_outcome.expected_net_saved_revenue:,.2f}`",
        f"- **Expected Campaign ROI**: `{baseline_outcome.expected_campaign_roi_pct:.1f}%`",
        "",
        "---",
        "",
        "## 📊 What-If Scenario Comparison",
        "",
        "Comparing portfolio outcomes across varying market conditions and retention assumptions:",
        "",
    ])

    comp_table = comparison_df[[
        "scenario_name",
        "budget_spent",
        "customers_targeted",
        "expected_customers_saved",
        "expected_gross_revenue_saved",
        "expected_net_saved_revenue",
        "expected_campaign_roi_pct",
        "cost_per_saved_customer",
    ]].copy()
    comp_table.columns = [
        "Scenario",
        "Spend ($)",
        "Targeted",
        "Exp Saved",
        "Gross Saved ($)",
        "Net Saved ($)",
        "ROI (%)",
        "Cost/Saved ($)",
    ]
    md_lines.append(_df_to_markdown(comp_table))
    md_lines.append("")

    # Section 5: Budget Sensitivity
    md_lines.extend([
        "---",
        "",
        "## 📈 Budget Sensitivity & Diminishing Marginal Returns",
        "",
        "Sweeping campaign budget from $1,000 to $45,000 reveals the efficiency frontier of the customer pool:",
        "",
    ])

    sweep_table = budget_sweep_df[[
        "budget_allocated",
        "budget_spent",
        "customers_targeted",
        "expected_customers_saved",
        "expected_net_saved_revenue",
        "expected_campaign_roi_pct",
    ]].copy()
    sweep_table.columns = [
        "Allocated Budget ($)",
        "Actual Spend ($)",
        "Targeted",
        "Exp Saved",
        "Net Savings ($)",
        "ROI (%)",
    ]
    md_lines.append(_df_to_markdown(sweep_table))
    md_lines.append("")

    md_lines.extend([
        "### Key Economic Takeaways:",
        "1. **Diminishing Marginal ROI**: As budget expands to lower-risk customer deciles, ROI drops from high double digits down to single digits.",
        "2. **Rational Saturation Cutoff**: When targeting is constrained by $\\mathbb{E}[\\text{Net Value}] > 0$, spend naturally caps once all economically viable at-risk accounts are targeted, preventing wasteful budget dissipation.",
        "3. **Artifact Visualizations**: High-resolution 3-panel visualization available at `artifacts/figures/18_retention_budget_scenarios.png`.",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Retention Budget Simulator")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BASE_DIR / "data" / "processed" / "telco_churn_clean.parquet",
        help="Path to clean customer dataset",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=BASE_DIR / "artifacts" / "models" / "calibrated_xgb_pipeline.joblib",
        help="Path to calibrated model pipeline",
    )
    parser.add_argument("--budget", type=float, default=10000.0, help="Retention campaign budget ($)")
    parser.add_argument("--cost-per-offer", type=float, default=35.0, help="Marginal cost per offer ($)")
    parser.add_argument("--offer-success-rate", type=float, default=0.22, help="Assumed offer success rate")
    parser.add_argument("--fixed-clv", type=float, default=450.0, help="Baseline CLV ($)")
    parser.add_argument("--clv-mode", type=str, default="fixed", choices=["fixed", "customer_specific", "hybrid"])
    parser.add_argument("--full-cohort", action="store_true", help="Evaluate full customer base instead of test split only")

    args = parser.parse_args()

    run_simulator_pipeline(
        dataset_path=args.dataset,
        model_path=args.model_path,
        budget=args.budget,
        cost_per_offer=args.cost_per_offer,
        offer_success_rate=args.offer_success_rate,
        fixed_clv=args.fixed_clv,
        clv_mode_str=args.clv_mode,
        use_test_partition_only=not args.full_cohort,
    )


if __name__ == "__main__":
    main()
