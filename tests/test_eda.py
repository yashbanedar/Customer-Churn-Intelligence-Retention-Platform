"""
Unit tests for Phase 3 Exploratory Data Analysis (EDA) module:
Tests statistical analyzer, hypothesis testing, visualization generation,
and report compilation.
"""

import json
from pathlib import Path
import pytest

from src.eda.analyzer import EDAAnalyzer
from src.eda.report_generator import EDAReportGenerator
from src.eda.run_eda import run_eda
from src.eda.visualizer import EDAVisualizer


@pytest.fixture
def analyzer() -> EDAAnalyzer:
    return EDAAnalyzer()


def test_analyze_target_balance(analyzer):
    """
    Verifies target balance calculations, churn rate %, and imbalance ratio.
    """
    tb = analyzer.analyze_target_balance()
    assert tb["total_customers"] == 7043
    assert tb["churned_count"] == 1869
    assert tb["retained_count"] == 5174
    assert tb["churn_rate_pct"] == 26.54
    assert tb["imbalance_ratio"] == 2.77
    assert "PR-AUC" in tb["evaluation_implication"]


def test_analyze_numeric_features(analyzer):
    """
    Verifies Mann-Whitney U test calculations and Rank-Biserial effect sizes.
    """
    num_results = analyzer.analyze_numeric_features()
    assert "tenure" in num_results
    assert "monthly_charges" in num_results
    assert "total_charges" in num_results

    # Tenure: Churned customers have significantly lower median tenure
    tenure_res = num_results["tenure"]
    assert tenure_res.churn_median < tenure_res.non_churn_median
    assert tenure_res.p_value < 1e-10
    assert tenure_res.rank_biserial_r < 0  # Lower in churned

    # Monthly charges: Churned customers have significantly higher median monthly charges
    charges_res = num_results["monthly_charges"]
    assert charges_res.churn_median > charges_res.non_churn_median
    assert charges_res.p_value < 1e-10
    assert charges_res.rank_biserial_r > 0  # Higher in churned


def test_analyze_categorical_features(analyzer):
    """
    Verifies Chi-Square contingency tests and Cramér's V ranking.
    """
    cat_results = analyzer.analyze_categorical_features()
    assert len(cat_results) >= 15
    # Contract should be the highest association with Cramér's V > 0.35
    top_feature = list(cat_results.keys())[0]
    assert top_feature == "contract"
    assert cat_results["contract"].cramers_v > 0.35
    assert cat_results["contract"].levels_churn_rates["Month-to-month"] > 40.0
    assert cat_results["contract"].levels_churn_rates["Two year"] < 5.0


def test_analyze_service_feature_stacking(analyzer):
    """
    Verifies add-on product stacking calculation (0 to 6 add-ons).
    """
    stack_df = analyzer.analyze_service_feature_stacking()
    assert len(stack_df) == 7  # 0, 1, 2, 3, 4, 5, 6 add-ons
    assert stack_df["customer_count"].sum() == 7043

    # Customers with 6 add-ons must have lowest churn rate
    churn_6 = stack_df[stack_df["addon_count"] == 6]["churn_rate_pct"].values[0]
    churn_1 = stack_df[stack_df["addon_count"] == 1]["churn_rate_pct"].values[0]
    assert churn_6 < 10.0
    assert churn_1 > 40.0


def test_assess_target_leakage(analyzer):
    """
    Verifies target leakage audit: total_charges collinearity check and leakage suspects.
    """
    leakage = analyzer.assess_target_leakage()
    assert leakage["leakage_suspects_count"] == 0
    assert leakage["total_charges_formula_correlation"] > 0.95
    assert len(leakage["detailed_findings"]) >= 4


def test_visualizer_generates_all_figures(analyzer, tmp_path):
    """
    Verifies that EDAVisualizer renders all 7 static figures and 1 interactive HTML dashboard.
    """
    visualizer = EDAVisualizer(df=analyzer.df, output_dir=tmp_path)
    rendered = visualizer.render_all()

    assert len(rendered) == 8
    expected_keys = [
        "target_distribution",
        "tenure_dynamics",
        "monthly_charges",
        "contract_churn",
        "services_addons",
        "payment_friction",
        "correlation_matrix",
        "interactive_html",
    ]
    for key in expected_keys:
        assert key in rendered
        p = rendered[key]
        assert p.exists()
        assert p.stat().st_size > 0


def test_report_generator(analyzer, tmp_path):
    """
    Verifies that EDAReportGenerator writes valid JSON and Markdown reports.
    """
    generator = EDAReportGenerator(analyzer=analyzer, output_dir=tmp_path)
    report_data = generator.generate_report()

    assert isinstance(report_data, dict)
    assert report_data["metadata"]["total_records"] == 7043

    # Check JSON
    json_path = tmp_path / "eda_summary.json"
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["metadata"]["total_records"] == 7043

    # Check Markdown
    md_path = tmp_path / "eda_findings_report.md"
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "Comprehensive Exploratory Data Analysis" in content
    assert "Cramér's V" in content
    assert "Mann-Whitney U" in content
    assert "Association vs. Causation" in content


def test_run_eda_end_to_end(tmp_path):
    """
    Verifies run_eda CLI execution with isolated output directories.
    """
    fig_dir = tmp_path / "test_figures"
    metrics_dir = tmp_path / "test_metrics"
    success = run_eda(output_figures_dir=fig_dir, output_metrics_dir=metrics_dir)

    assert success is True
    assert (fig_dir / "01_target_distribution.png").exists()
    assert (metrics_dir / "eda_findings_report.md").exists()
    assert (metrics_dir / "eda_summary.json").exists()
