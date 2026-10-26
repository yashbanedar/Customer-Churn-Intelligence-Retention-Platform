"""
Unit tests for Streamlit Dashboard Helper Utilities (Phase 7).
Tests KPI computations, data formatting, filtering logic, and interactive simulation helpers.
"""

import numpy as np
import pandas as pd
import pytest

from app.helpers import (
    compute_portfolio_kpis,
    filter_customer_records,
    format_currency,
    format_percentage,
    load_live_database_summary,
    run_interactive_retention_simulation,
)


@pytest.fixture
def sample_kpi_dataframe():
    """Deterministic customer DataFrame for KPI and filtering tests."""
    return pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "monthly_charges": [100.0, 50.0, 50.0, 200.0],
        "churn": ["Yes", "No", "No", "Yes"],
        "contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "internet_service": ["Fiber optic", "DSL", "No", "Fiber optic"],
        "predicted_risk_tier": ["High Risk", "Low Risk", "Low Risk", "High Risk"],
    })


class TestFormattingHelpers:
    """Tests string formatters for currency and percentages."""

    def test_format_currency_positive(self):
        assert format_currency(1234.567) == "$1,234.57"
        assert format_currency(50) == "$50.00"

    def test_format_currency_nan_none(self):
        assert format_currency(None) == "$0.00"
        assert format_currency(float("nan")) == "$0.00"

    def test_format_percentage(self):
        assert format_percentage(26.543, 1) == "26.5%"
        assert format_percentage(26.543, 2) == "26.54%"
        assert format_percentage(None) == "0.0%"


class TestPortfolioKPICalculation:
    """Tests KPI aggregations across customer portfolios."""

    def test_compute_portfolio_kpis_empty(self):
        empty_df = pd.DataFrame()
        res = compute_portfolio_kpis(empty_df)
        assert res["total_customers"] == 0
        assert res["churn_rate_pct"] == 0.0
        assert res["total_mrr"] == 0.0

    def test_compute_portfolio_kpis_sample(self, sample_kpi_dataframe):
        res = compute_portfolio_kpis(sample_kpi_dataframe)
        assert res["total_customers"] == 4
        assert res["churned_customers"] == 2
        assert res["churn_rate_pct"] == 50.0
        assert res["total_mrr"] == 400.0
        assert res["mrr_at_risk"] == 300.0  # C1 ($100) + C4 ($200)
        assert res["avg_revenue_per_user"] == 100.0


class TestCustomerRecordFiltering:
    """Tests multi-criteria filtering on customer DataFrames."""

    def test_filter_by_risk_tier(self, sample_kpi_dataframe):
        res = filter_customer_records(sample_kpi_dataframe, risk_filter="High Risk")
        assert len(res) == 2
        assert set(res["customer_id"]) == {"C1", "C4"}

    def test_filter_by_contract(self, sample_kpi_dataframe):
        res = filter_customer_records(sample_kpi_dataframe, contract_filter="One year")
        assert len(res) == 1
        assert res.iloc[0]["customer_id"] == "C2"

    def test_filter_by_search_query(self, sample_kpi_dataframe):
        res = filter_customer_records(sample_kpi_dataframe, search_query="C3")
        assert len(res) == 1
        assert res.iloc[0]["customer_id"] == "C3"


class TestInteractiveSimulationHelper:
    """Tests real-time interactive retention simulation helper."""

    def test_run_interactive_simulation(self, sample_kpi_dataframe):
        probs = np.array([0.75, 0.15, 0.10, 0.85])
        outcome, res_df = run_interactive_retention_simulation(
            customers_df=sample_kpi_dataframe,
            probabilities=probs,
            budget=100.0,
            cost_per_offer=35.0,
            assumed_save_rate=0.22,
            fixed_clv=450.0,
            require_positive_ev=True,
        )
        assert outcome.budget_allocated == 100.0
        assert outcome.budget_spent <= 100.0
        assert outcome.customers_targeted >= 1
        assert len(res_df) == 4
        assert "simulated_is_targeted" in res_df.columns
