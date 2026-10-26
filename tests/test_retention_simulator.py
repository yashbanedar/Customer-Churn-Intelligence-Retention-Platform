"""
Unit and boundary tests for the Transparent Retention-Budget Simulator (Phase 6).
Verifies:
  1. Boundary constraints (zero budget, sub-offer budget, massive budget, zero/perfect success rates).
  2. Input validation and assumption checking for expected value ranking.
  3. Exact mathematical calculation tests on known mock portfolios.
  4. Customer-specific CLV arithmetic.
  5. Four-way transparent data segmentation (observed, predicted, assumed, simulated).
  6. Strict causal disclaimer compliance.
"""

import numpy as np
import pandas as pd
import pytest

from src.business.retention_simulator import (
    CAUSAL_DISCLAIMER,
    CLVCalculationMode,
    RankingStrategy,
    RetentionAssumptionConfig,
    RetentionBudgetSimulator,
    SimulationOutcome,
)


@pytest.fixture
def sample_mock_portfolio():
    """
    Creates a deterministic 3-customer synthetic cohort with hand-calculated expected values.
    Customer 1: High risk, large CLV -> High positive EV
    Customer 2: Moderate risk -> Small positive EV
    Customer 3: Low risk -> Negative EV
    """
    df = pd.DataFrame({
        "customerID": ["CUST_A", "CUST_B", "CUST_C"],
        "MonthlyCharges": [100.0, 50.0, 25.0],
        "tenure": [3, 12, 48],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "TotalCharges": [300.0, 600.0, 1200.0],
        "Churn": ["Yes", "No", "No"],
    })
    # Probabilities: 0.80, 0.50, 0.20
    probs = np.array([0.80, 0.50, 0.20])
    return df, probs


class TestRetentionAssumptionConfigValidation:
    """Tests boundary validation and error handling for configuration assumptions."""

    def test_default_config_valid(self):
        cfg = RetentionAssumptionConfig()
        cfg.validate()
        assert cfg.budget == 10000.0
        assert cfg.cost_per_offer == 35.0
        assert cfg.assumed_offer_success_rate == 0.22

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError, match="Budget must be non-negative"):
            RetentionAssumptionConfig(budget=-100.0)

    def test_zero_or_negative_offer_cost_raises(self):
        with pytest.raises(ValueError, match="Cost per offer must be strictly positive"):
            RetentionAssumptionConfig(cost_per_offer=0.0)
        with pytest.raises(ValueError, match="Cost per offer must be strictly positive"):
            RetentionAssumptionConfig(cost_per_offer=-25.0)

    def test_invalid_success_rate_raises(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            RetentionAssumptionConfig(assumed_offer_success_rate=-0.05)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            RetentionAssumptionConfig(assumed_offer_success_rate=1.25)

    def test_invalid_fixed_clv_raises(self):
        with pytest.raises(ValueError, match="Fixed CLV must be strictly positive"):
            RetentionAssumptionConfig(fixed_clv=0.0)
        with pytest.raises(ValueError, match="Fixed CLV must be strictly positive"):
            RetentionAssumptionConfig(fixed_clv=-400.0)

    def test_invalid_profit_margin_raises(self):
        with pytest.raises(ValueError, match="Profit margin must be in"):
            RetentionAssumptionConfig(profit_margin=0.0)
        with pytest.raises(ValueError, match="Profit margin must be in"):
            RetentionAssumptionConfig(profit_margin=1.5)


class TestRetentionSimulatorCalculations:
    """Tests exact hand-calculated financial arithmetic on synthetic portfolios."""

    def test_exact_expected_value_and_ranking(self, sample_mock_portfolio):
        """
        Verify exact arithmetic:
        Assumptions: Cost C = $40, Save Rate r = 0.20, Fixed CLV V = $500
        Customer A: p = 0.80 -> Gross = 0.80 * 0.20 * 500 = $80.00, Net = 80 - 40 = $40.00, ROI = 100%
        Customer B: p = 0.50 -> Gross = 0.50 * 0.20 * 500 = $50.00, Net = 50 - 40 = $10.00, ROI = 25%
        Customer C: p = 0.20 -> Gross = 0.20 * 0.20 * 500 = $20.00, Net = 20 - 40 = -$20.00, ROI = -50%
        """
        df, probs = sample_mock_portfolio
        cfg = RetentionAssumptionConfig(
            scenario_name="ExactMathTest",
            budget=100.0,
            cost_per_offer=40.0,
            assumed_offer_success_rate=0.20,
            clv_mode=CLVCalculationMode.FIXED,
            fixed_clv=500.0,
            ranking_strategy=RankingStrategy.EXPECTED_NET_VALUE,
            require_positive_expected_value=True,
        )
        sim = RetentionBudgetSimulator(config=cfg)
        outcome, res_df = sim.simulate(df, probs, config=cfg)

        # Check ranking order
        assert res_df.iloc[0]["observed_customer_id"] == "CUST_A"
        assert res_df.iloc[1]["observed_customer_id"] == "CUST_B"
        assert res_df.iloc[2]["observed_customer_id"] == "CUST_C"

        # Check individual expected net values
        assert pytest.approx(res_df.iloc[0]["simulated_expected_net_value"], rel=1e-3) == 40.00
        assert pytest.approx(res_df.iloc[1]["simulated_expected_net_value"], rel=1e-3) == 10.00
        assert pytest.approx(res_df.iloc[2]["simulated_expected_net_value"], rel=1e-3) == -20.00

        # Check targeting under Budget = $100 (Can afford 2 offers at $40 each = $80 spend, $20 remaining)
        assert res_df.iloc[0]["simulated_is_targeted"] == True
        assert res_df.iloc[1]["simulated_is_targeted"] == True
        assert res_df.iloc[2]["simulated_is_targeted"] == False

        # Check aggregate outcome metrics
        assert outcome.customers_targeted == 2
        assert outcome.budget_spent == 80.0
        assert outcome.budget_remaining == 20.0
        assert pytest.approx(outcome.expected_customers_saved, rel=1e-3) == (0.80 * 0.20) + (0.50 * 0.20)  # 0.26
        assert pytest.approx(outcome.expected_gross_revenue_saved, rel=1e-3) == 80.0 + 50.0  # $130.00
        assert pytest.approx(outcome.expected_net_saved_revenue, rel=1e-3) == 130.0 - 80.0  # $50.00
        assert pytest.approx(outcome.expected_campaign_roi_pct, rel=1e-3) == (50.0 / 80.0) * 100.0  # 62.5%

    def test_customer_specific_clv_calculation(self):
        """
        Verify customer-specific CLV arithmetic:
        Customer: MonthlyCharges = $100, Horizon = 12 months, Margin = 0.80
        CLV = 100 * 12 * 0.80 = $960.00
        p = 0.50, r = 0.25, C = $35.00
        Gross = 0.50 * 0.25 * 960 = $120.00
        Net = 120 - 35 = $85.00
        """
        df = pd.DataFrame({
            "customer_id": ["VIP_001"],
            "monthly_charges": [100.0],
            "tenure": [10],
            "contract": ["Month-to-month"],
        })
        probs = np.array([0.50])
        cfg = RetentionAssumptionConfig(
            budget=100.0,
            cost_per_offer=35.0,
            assumed_offer_success_rate=0.25,
            clv_mode=CLVCalculationMode.CUSTOMER_SPECIFIC,
            clv_horizon_months=12,
            profit_margin=0.80,
            ranking_strategy=RankingStrategy.EXPECTED_NET_VALUE,
        )
        sim = RetentionBudgetSimulator(config=cfg)
        outcome, res_df = sim.simulate(df, probs, config=cfg)

        assert pytest.approx(res_df.iloc[0]["assumed_customer_clv"], rel=1e-3) == 960.00
        assert pytest.approx(res_df.iloc[0]["simulated_expected_gross_saved"], rel=1e-3) == 120.00
        assert pytest.approx(res_df.iloc[0]["simulated_expected_net_value"], rel=1e-3) == 85.00
        assert outcome.customers_targeted == 1
        assert outcome.budget_spent == 35.0
        assert outcome.expected_net_saved_revenue == 85.00


class TestRetentionSimulatorBoundaryConditions:
    """Tests extreme boundary edges: zero budget, tiny budget, infinite budget, zero/perfect success rates."""

    def test_zero_budget_targets_none(self, sample_mock_portfolio):
        df, probs = sample_mock_portfolio
        cfg = RetentionAssumptionConfig(budget=0.0, cost_per_offer=35.0)
        sim = RetentionBudgetSimulator(config=cfg)
        outcome, res_df = sim.simulate(df, probs, config=cfg)

        assert outcome.customers_targeted == 0
        assert outcome.budget_spent == 0.0
        assert outcome.budget_remaining == 0.0
        assert outcome.expected_customers_saved == 0.0
        assert outcome.expected_gross_revenue_saved == 0.0
        assert outcome.expected_net_saved_revenue == 0.0
        assert outcome.expected_campaign_roi_pct == 0.0
        assert not res_df["simulated_is_targeted"].any()

    def test_budget_smaller_than_single_offer(self, sample_mock_portfolio):
        df, probs = sample_mock_portfolio
        # Budget $20 cannot afford $35 offer
        cfg = RetentionAssumptionConfig(budget=20.0, cost_per_offer=35.0)
        sim = RetentionBudgetSimulator(config=cfg)
        outcome, res_df = sim.simulate(df, probs, config=cfg)

        assert outcome.customers_targeted == 0
        assert outcome.budget_spent == 0.0
        assert outcome.budget_remaining == 20.0
        assert not res_df["simulated_is_targeted"].any()

    def test_massive_budget_respects_positive_ev_guardrail(self, sample_mock_portfolio):
        """
        With $1,000,000 budget and require_positive_expected_value=True,
        simulator MUST NOT target Customer C because Customer C has negative EV.
        """
        df, probs = sample_mock_portfolio
        cfg = RetentionAssumptionConfig(
            budget=1_000_000.0,
            cost_per_offer=40.0,
            assumed_offer_success_rate=0.20,
            fixed_clv=500.0,
            require_positive_expected_value=True,
        )
        sim = RetentionBudgetSimulator(config=cfg)
        outcome, res_df = sim.simulate(df, probs, config=cfg)

        # Only CUST_A and CUST_B have positive EV
        assert outcome.customers_targeted == 2
        assert outcome.budget_spent == 80.0
        assert outcome.budget_remaining == 999_920.0
        assert res_df.loc[res_df["observed_customer_id"] == "CUST_C", "simulated_is_targeted"].values[0] == False

    def test_massive_budget_targets_all_when_guardrail_disabled(self, sample_mock_portfolio):
        """
        With require_positive_expected_value=False, all candidates up to pool size can be targeted.
        """
        df, probs = sample_mock_portfolio
        cfg = RetentionAssumptionConfig(
            budget=1_000_000.0,
            cost_per_offer=40.0,
            assumed_offer_success_rate=0.20,
            fixed_clv=500.0,
            require_positive_expected_value=False,
        )
        sim = RetentionBudgetSimulator(config=cfg)
        outcome, res_df = sim.simulate(df, probs, config=cfg)

        assert outcome.customers_targeted == 3
        assert outcome.budget_spent == 120.0
        assert res_df["simulated_is_targeted"].all()

    def test_zero_success_rate_boundary(self, sample_mock_portfolio):
        """
        When offer_success_rate = 0.0:
        E[Gross] = 0 for all customers.
        E[Net] = -C for all customers.
        Under require_positive_expected_value=True, 0 customers must be targeted.
        """
        df, probs = sample_mock_portfolio
        cfg = RetentionAssumptionConfig(
            budget=1000.0,
            cost_per_offer=35.0,
            assumed_offer_success_rate=0.0,
            require_positive_expected_value=True,
        )
        sim = RetentionBudgetSimulator(config=cfg)
        outcome, res_df = sim.simulate(df, probs, config=cfg)

        assert outcome.customers_targeted == 0
        assert outcome.budget_spent == 0.0
        assert outcome.expected_net_saved_revenue == 0.0

    def test_prohibitive_offer_cost_boundary(self, sample_mock_portfolio):
        """
        If offer cost ($1,000) exceeds maximum CLV ($500), EV is negative for all accounts.
        Zero customers should be targeted.
        """
        df, probs = sample_mock_portfolio
        cfg = RetentionAssumptionConfig(
            budget=5000.0,
            cost_per_offer=1000.0,
            assumed_offer_success_rate=0.22,
            fixed_clv=500.0,
            require_positive_expected_value=True,
        )
        sim = RetentionBudgetSimulator(config=cfg)
        outcome, _ = sim.simulate(df, probs, config=cfg)

        assert outcome.customers_targeted == 0
        assert outcome.budget_spent == 0.0


class TestDataSegmentationAndCausalCompliance:
    """Verifies four-way transparent column classification and causal disclaimers."""

    def test_data_segmentation_columns(self, sample_mock_portfolio):
        df, probs = sample_mock_portfolio
        sim = RetentionBudgetSimulator()
        _, res_df = sim.simulate(df, probs)

        cols = list(res_df.columns)
        # 1. Observed Data
        assert any(c.startswith("observed_") for c in cols)
        assert "observed_customer_id" in cols
        assert "observed_monthly_charges" in cols
        # 2. Model Prediction
        assert any(c.startswith("predicted_") for c in cols)
        assert "predicted_calibrated_churn_prob" in cols
        assert "predicted_risk_tier" in cols
        # 3. Hypothetical Assumption
        assert any(c.startswith("assumed_") for c in cols)
        assert "assumed_offer_cost" in cols
        assert "assumed_success_rate" in cols
        assert "assumed_is_counterfactual_scenario" in cols
        # 4. Simulated Outcome
        assert any(c.startswith("simulated_") for c in cols)
        assert "simulated_expected_gross_saved" in cols
        assert "simulated_expected_net_value" in cols
        assert "simulated_is_targeted" in cols
        assert "simulated_target_priority_rank" in cols

    def test_strict_causal_disclaimer_present(self, sample_mock_portfolio):
        df, probs = sample_mock_portfolio
        sim = RetentionBudgetSimulator()
        outcome, _ = sim.simulate(df, probs)

        assert outcome.causal_disclaimer == CAUSAL_DISCLAIMER
        assert "counterfactual" in outcome.causal_disclaimer.lower()
        assert "randomized" in outcome.causal_disclaimer.lower()
        assert "not verified causal treatment effects" in outcome.causal_disclaimer.lower()
