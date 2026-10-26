"""
Transparent Retention-Budget Simulator.
Models campaign costs, customer selection under budget constraints,
expected value optimization, and counterfactual scenario analysis.

CRITICAL METHODOLOGICAL PRINCIPLE:
This simulator explicitly segregates:
  1. [OBSERVED DATA]: Empirical attributes from customer records (tenure, charges, contract).
  2. [MODEL PREDICTION]: Calibrated churn probabilities (statistical associations under status-quo).
  3. [HYPOTHETICAL ASSUMPTION]: User-defined scenarios (budget, offer cost, assumed offer success rate, CLV).
  4. [SIMULATED OUTCOME]: Expected value, ROI, and budget allocation under specified assumptions.

NO CAUSAL RETENTION UPLIFT IS CLAIMED WITHOUT INTERVENTION / RCT DATA.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BASE_DIR, get_settings


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


CAUSAL_DISCLAIMER: str = (
    "CAUSAL RETENTION UPLIFT DISCLAIMER: The retention uplift, saved customers, and "
    "expected revenue figures in this simulation are strictly computed from user-specified "
    "hypothetical assumptions (assumed offer success rate r and offer cost C). "
    "The underlying dataset contains observational records without randomized intervention "
    "or A/B test data. These metrics represent counterfactual scenario projections under "
    "status-quo predictive models, NOT verified causal treatment effects (CATE). "
    "Empirical verification requires running a randomized controlled trial."
)


class CLVCalculationMode(str, Enum):
    """Modes for customer lifetime value (CLV) evaluation."""
    FIXED = "fixed"
    CUSTOMER_SPECIFIC = "customer_specific"
    HYBRID = "hybrid"


class RankingStrategy(str, Enum):
    """Customer ranking strategies for retention targeting."""
    EXPECTED_NET_VALUE = "expected_net_value"
    EXPECTED_ROI = "expected_roi"
    CALIBRATED_PROBABILITY = "calibrated_probability"


@dataclass
class RetentionAssumptionConfig:
    """
    Explicit parameters and assumptions for a retention budget simulation scenario.

    Attributes:
        scenario_name: Name of the scenario (e.g. 'Conservative', 'Baseline', 'Optimistic').
        budget: Total retention campaign budget in currency units (e.g. USD). Must be >= 0.
        cost_per_offer: Marginal direct cost per proactive customer intervention. Must be > 0.
        assumed_offer_success_rate: Assumed probability that an offer prevents churn if customer
                                    was indeed going to churn (r in [0.0, 1.0]).
        clv_mode: CLV computation mode ('fixed', 'customer_specific', or 'hybrid').
        fixed_clv: Baseline net CLV per retained customer (used in fixed or hybrid mode).
        clv_horizon_months: Planning horizon for customer-specific CLV (months saved).
        profit_margin: Operating gross profit margin multiplier on monthly charges (0.0 < margin <= 1.0).
        ranking_strategy: Strategy to rank customers ('expected_net_value', 'expected_roi', 'calibrated_probability').
        require_positive_expected_value: If True, only targets customers where E[Net Value] > 0.
    """
    scenario_name: str = "Baseline"
    budget: float = 10000.0
    cost_per_offer: float = 35.0
    assumed_offer_success_rate: float = 0.22
    clv_mode: CLVCalculationMode = CLVCalculationMode.FIXED
    fixed_clv: float = 450.0
    clv_horizon_months: int = 12
    profit_margin: float = 0.80
    ranking_strategy: RankingStrategy = RankingStrategy.EXPECTED_NET_VALUE
    require_positive_expected_value: bool = True

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validates numerical ranges and constraints on business assumptions."""
        if self.budget < 0.0:
            raise ValueError(f"Budget must be non-negative. Received: {self.budget}")
        if self.cost_per_offer <= 0.0:
            raise ValueError(f"Cost per offer must be strictly positive. Received: {self.cost_per_offer}")
        if not (0.0 <= self.assumed_offer_success_rate <= 1.0):
            raise ValueError(
                f"Assumed offer success rate must be between 0.0 and 1.0. Received: {self.assumed_offer_success_rate}"
            )
        if self.fixed_clv <= 0.0:
            raise ValueError(f"Fixed CLV must be strictly positive. Received: {self.fixed_clv}")
        if self.clv_horizon_months <= 0:
            raise ValueError(f"CLV horizon months must be positive integer. Received: {self.clv_horizon_months}")
        if not (0.0 < self.profit_margin <= 1.0):
            raise ValueError(f"Profit margin must be in (0.0, 1.0]. Received: {self.profit_margin}")

    def validate_for_ev_ranking(self) -> None:
        """
        Ensures all necessary inputs and assumptions are explicitly defined
        before allowing expected value ranking.
        """
        self.validate()
        if self.ranking_strategy in (RankingStrategy.EXPECTED_NET_VALUE, RankingStrategy.EXPECTED_ROI):
            if self.assumed_offer_success_rate is None or self.cost_per_offer is None:
                raise ValueError(
                    "Expected Value ranking requires explicit assumed_offer_success_rate and cost_per_offer."
                )


@dataclass
class SimulationOutcome:
    """
    Aggregated performance outcome and financial metrics from a single simulation run.
    """
    scenario_name: str
    budget_allocated: float
    budget_spent: float
    budget_remaining: float
    budget_utilization_pct: float
    total_candidate_pool: int
    customers_targeted: int
    targeting_rate_pct: float
    expected_customers_saved: float
    expected_gross_revenue_saved: float
    total_campaign_cost: float
    expected_net_saved_revenue: float
    expected_campaign_roi_pct: float
    cost_per_saved_customer: float
    cost_per_offer: float
    assumed_offer_success_rate: float
    clv_mode: str
    ranking_strategy: str
    require_positive_ev: bool
    causal_disclaimer: str = field(default=CAUSAL_DISCLAIMER)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes outcome metrics to dictionary."""
        return asdict(self)


class RetentionBudgetSimulator:
    """
    Transparent retention campaign simulator for customer prioritization,
    budget allocation, and what-if financial sensitivity analysis.
    """

    def __init__(self, config: Optional[RetentionAssumptionConfig] = None):
        self.config = config or RetentionAssumptionConfig()
        self.config.validate()

    def compute_customer_clv(
        self,
        df: pd.DataFrame,
        monthly_charges_col: str = "MonthlyCharges",
        config: Optional[RetentionAssumptionConfig] = None,
    ) -> np.ndarray:
        """
        Computes customer lifetime value (CLV) based on configured mode.

        Args:
            df: Customer DataFrame containing monthly charges.
            monthly_charges_col: Column name for monthly charges.
            config: Optional configuration override.

        Returns:
            1D array of customer values (CLV) in dollars.
        """
        cfg = config or self.config
        n_customers = len(df)
        if cfg.clv_mode == CLVCalculationMode.FIXED:
            return np.full(n_customers, float(cfg.fixed_clv))

        # Check for monthly charges
        col = None
        for candidate in ["MonthlyCharges", "monthly_charges", monthly_charges_col]:
            if candidate in df.columns:
                col = candidate
                break
        if col is not None:
            monthly_rev = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
        else:
            monthly_rev = np.full(n_customers, 65.0)  # Telco median fallback

        customer_specific_val = (
            monthly_rev * cfg.profit_margin * cfg.clv_horizon_months
        )

        if cfg.clv_mode == CLVCalculationMode.CUSTOMER_SPECIFIC:
            # Ensure minimum positive value
            return np.maximum(customer_specific_val, 10.0)

        if cfg.clv_mode == CLVCalculationMode.HYBRID:
            # Hybrid takes max of baseline fixed CLV and customer-specific estimate
            return np.maximum(customer_specific_val, float(cfg.fixed_clv))

        return np.full(n_customers, float(cfg.fixed_clv))

    def simulate(
        self,
        customers_df: pd.DataFrame,
        churn_probabilities: np.ndarray,
        config: Optional[RetentionAssumptionConfig] = None,
    ) -> Tuple[SimulationOutcome, pd.DataFrame]:
        """
        Executes a transparent retention budget simulation.

        Steps:
          1. Validates inputs and explicit assumptions.
          2. Computes Expected Gross Saved Value and Expected Net Value per customer.
          3. Ranks customers according to explicit ranking strategy.
          4. Allocates budget sequentially until budget is exhausted.
          5. Segregates output into Observed Data, Model Predictions,
             Hypothetical Assumptions, and Simulated Outcomes.

        Args:
            customers_df: DataFrame of customer records.
            churn_probabilities: 1D array of calibrated churn probabilities.
            config: Optional custom scenario configuration.

        Returns:
            Tuple of (SimulationOutcome, detailed_segmented_dataframe).
        """
        cfg = config or self.config
        cfg.validate_for_ev_ranking()

        if len(customers_df) != len(churn_probabilities):
            raise ValueError(
                f"Length mismatch: customers_df ({len(customers_df)}) vs "
                f"churn_probabilities ({len(churn_probabilities)})."
            )

        n_customers = len(customers_df)
        probs = np.asarray(churn_probabilities, dtype=float)
        probs = np.clip(probs, 0.0, 1.0)

        # 1. Compute customer CLV under explicit assumption
        clv_array = self.compute_customer_clv(customers_df, config=cfg)

        # 2. Compute individual expected financial outcomes
        # E[Gross Saved Value] = P(Churn) * Assumed_Success_Rate * CLV
        exp_gross_saved = probs * cfg.assumed_offer_success_rate * clv_array
        # E[Net Value] = E[Gross Saved Value] - Cost_Per_Offer
        exp_net_value = exp_gross_saved - cfg.cost_per_offer
        # E[ROI %] = (E[Net Value] / Cost_Per_Offer) * 100
        exp_roi_pct = (exp_net_value / cfg.cost_per_offer) * 100.0

        # 3. Create transparent segmented representation
        # Explicit column prefixes: observed_, predicted_, assumed_, simulated_
        res_df = pd.DataFrame()

        # [OBSERVED DATA]
        id_col = "customerID" if "customerID" in customers_df.columns else "customer_id"
        if id_col in customers_df.columns:
            res_df["observed_customer_id"] = customers_df[id_col].values
        else:
            res_df["observed_customer_id"] = [f"CUST_{i:05d}" for i in range(n_customers)]

        m_col = "monthly_charges" if "monthly_charges" in customers_df.columns else "MonthlyCharges"
        res_df["observed_monthly_charges"] = (
            pd.to_numeric(customers_df[m_col], errors="coerce").fillna(0.0).values
            if m_col in customers_df.columns
            else np.nan
        )
        t_col = "tenure" if "tenure" in customers_df.columns else "Tenure"
        res_df["observed_tenure_months"] = (
            pd.to_numeric(customers_df[t_col], errors="coerce").fillna(0).values
            if t_col in customers_df.columns
            else np.nan
        )
        c_col = "contract" if "contract" in customers_df.columns else "Contract"
        res_df["observed_contract_type"] = (
            customers_df[c_col].values if c_col in customers_df.columns else "Unknown"
        )
        tot_col = "total_charges" if "total_charges" in customers_df.columns else "TotalCharges"
        res_df["observed_total_charges"] = (
            pd.to_numeric(customers_df[tot_col], errors="coerce").fillna(0.0).values
            if tot_col in customers_df.columns
            else np.nan
        )
        for ch_col in ["churn_label", "churn", "Churn"]:
            if ch_col in customers_df.columns:
                res_df["observed_actual_churn"] = customers_df[ch_col].values
                break

        # [MODEL PREDICTION]
        res_df["predicted_calibrated_churn_prob"] = np.round(probs, 4)
        risk_tiers = np.where(
            probs >= 0.60, "High Risk", np.where(probs >= 0.30, "Medium Risk", "Low Risk")
        )
        res_df["predicted_risk_tier"] = risk_tiers

        # [HYPOTHETICAL ASSUMPTION]
        res_df["assumed_offer_cost"] = float(cfg.cost_per_offer)
        res_df["assumed_success_rate"] = float(cfg.assumed_offer_success_rate)
        res_df["assumed_customer_clv"] = np.round(clv_array, 2)
        res_df["assumed_is_counterfactual_scenario"] = True

        # [SIMULATED OUTCOME]
        res_df["simulated_expected_gross_saved"] = np.round(exp_gross_saved, 2)
        res_df["simulated_expected_net_value"] = np.round(exp_net_value, 2)
        res_df["simulated_expected_roi_pct"] = np.round(exp_roi_pct, 2)

        # 4. Rank candidates based on explicit strategy
        if cfg.ranking_strategy == RankingStrategy.EXPECTED_NET_VALUE:
            # Primary: Net EV, Secondary: Probability
            res_df = res_df.sort_values(
                by=["simulated_expected_net_value", "predicted_calibrated_churn_prob"],
                ascending=[False, False],
            ).reset_index(drop=True)
        elif cfg.ranking_strategy == RankingStrategy.EXPECTED_ROI:
            res_df = res_df.sort_values(
                by=["simulated_expected_roi_pct", "predicted_calibrated_churn_prob"],
                ascending=[False, False],
            ).reset_index(drop=True)
        elif cfg.ranking_strategy == RankingStrategy.CALIBRATED_PROBABILITY:
            res_df = res_df.sort_values(
                by=["predicted_calibrated_churn_prob"],
                ascending=[False],
            ).reset_index(drop=True)

        res_df["simulated_target_priority_rank"] = np.arange(1, n_customers + 1)

        # 5. Sequential budget targeting
        is_targeted = np.zeros(n_customers, dtype=bool)
        remaining_budget = float(cfg.budget)
        spend = 0.0

        for idx, row in res_df.iterrows():
            # Check economic rationality if configured
            if cfg.require_positive_expected_value and row["simulated_expected_net_value"] <= 0.0:
                continue

            # Check if budget accommodates another offer
            if remaining_budget >= cfg.cost_per_offer:
                is_targeted[idx] = True
                remaining_budget -= cfg.cost_per_offer
                spend += cfg.cost_per_offer
            else:
                # Cannot afford another offer
                break

        res_df["simulated_is_targeted"] = is_targeted

        # 6. Aggregate scenario outcomes
        targeted_df = res_df[res_df["simulated_is_targeted"]]
        n_targeted = len(targeted_df)
        total_campaign_cost = float(spend)
        budget_remaining = float(cfg.budget - total_campaign_cost)
        budget_util_pct = (
            float((total_campaign_cost / cfg.budget) * 100.0) if cfg.budget > 0 else 0.0
        )
        targeting_rate_pct = float((n_targeted / n_customers) * 100.0) if n_customers > 0 else 0.0

        if n_targeted > 0:
            exp_cust_saved = float(
                (targeted_df["predicted_calibrated_churn_prob"] * cfg.assumed_offer_success_rate).sum()
            )
            exp_gross_rev = float(targeted_df["simulated_expected_gross_saved"].sum())
            exp_net_rev = float(targeted_df["simulated_expected_net_value"].sum())
            campaign_roi = float((exp_net_rev / total_campaign_cost) * 100.0)
            cost_per_saved = float(total_campaign_cost / exp_cust_saved) if exp_cust_saved > 0 else 0.0
        else:
            exp_cust_saved = 0.0
            exp_gross_rev = 0.0
            exp_net_rev = 0.0
            campaign_roi = 0.0
            cost_per_saved = 0.0

        outcome = SimulationOutcome(
            scenario_name=cfg.scenario_name,
            budget_allocated=float(cfg.budget),
            budget_spent=float(total_campaign_cost),
            budget_remaining=float(budget_remaining),
            budget_utilization_pct=round(budget_util_pct, 2),
            total_candidate_pool=int(n_customers),
            customers_targeted=int(n_targeted),
            targeting_rate_pct=round(targeting_rate_pct, 2),
            expected_customers_saved=round(exp_cust_saved, 2),
            expected_gross_revenue_saved=round(exp_gross_rev, 2),
            total_campaign_cost=round(total_campaign_cost, 2),
            expected_net_saved_revenue=round(exp_net_rev, 2),
            expected_campaign_roi_pct=round(campaign_roi, 2),
            cost_per_saved_customer=round(cost_per_saved, 2),
            cost_per_offer=float(cfg.cost_per_offer),
            assumed_offer_success_rate=float(cfg.assumed_offer_success_rate),
            clv_mode=cfg.clv_mode.value if isinstance(cfg.clv_mode, CLVCalculationMode) else str(cfg.clv_mode),
            ranking_strategy=(
                cfg.ranking_strategy.value
                if isinstance(cfg.ranking_strategy, RankingStrategy)
                else str(cfg.ranking_strategy)
            ),
            require_positive_ev=bool(cfg.require_positive_expected_value),
        )

        logger.info(
            f"Simulation '{cfg.scenario_name}' Complete: "
            f"Targeted {n_targeted}/{n_customers} ({targeting_rate_pct:.1f}%), "
            f"Spend: ${total_campaign_cost:,.2f}/${cfg.budget:,.2f} ({budget_util_pct:.1f}%), "
            f"Exp Saved: {exp_cust_saved:.1f} accounts, "
            f"Exp Net Saved Revenue: ${exp_net_rev:,.2f} (ROI: {campaign_roi:.1f}%)"
        )

        return outcome, res_df

    def compare_scenarios(
        self,
        customers_df: pd.DataFrame,
        churn_probabilities: np.ndarray,
        scenarios: List[RetentionAssumptionConfig],
    ) -> pd.DataFrame:
        """
        Runs multiple what-if retention scenarios side-by-side.

        Args:
            customers_df: Customer DataFrame.
            churn_probabilities: Calibrated probabilities.
            scenarios: List of scenario configurations.

        Returns:
            Comparison summary DataFrame.
        """
        records = []
        for sc in scenarios:
            outcome, _ = self.simulate(customers_df, churn_probabilities, config=sc)
            records.append(outcome.to_dict())

        df_comp = pd.DataFrame(records)
        return df_comp

    def sweep_budget_sensitivity(
        self,
        customers_df: pd.DataFrame,
        churn_probabilities: np.ndarray,
        budgets: List[float],
        base_config: Optional[RetentionAssumptionConfig] = None,
    ) -> pd.DataFrame:
        """
        Sweeps campaign budget across multiple allocation levels to map the
        diminishing marginal returns retention frontier.

        Args:
            customers_df: Customer DataFrame.
            churn_probabilities: Calibrated churn probabilities.
            budgets: List of budget amounts.
            base_config: Base scenario parameters.

        Returns:
            DataFrame of financial outcomes across the budget spectrum.
        """
        base = base_config or self.config
        rows = []
        for b in sorted(budgets):
            cfg = RetentionAssumptionConfig(
                scenario_name=f"Budget_${b:,.0f}",
                budget=float(b),
                cost_per_offer=base.cost_per_offer,
                assumed_offer_success_rate=base.assumed_offer_success_rate,
                clv_mode=base.clv_mode,
                fixed_clv=base.fixed_clv,
                clv_horizon_months=base.clv_horizon_months,
                profit_margin=base.profit_margin,
                ranking_strategy=base.ranking_strategy,
                require_positive_expected_value=base.require_positive_expected_value,
            )
            outcome, _ = self.simulate(customers_df, churn_probabilities, config=cfg)
            rows.append(outcome.to_dict())

        return pd.DataFrame(rows)

    def plot_scenario_analysis(
        self,
        comparison_df: pd.DataFrame,
        budget_sweep_df: pd.DataFrame,
        target_records_df: pd.DataFrame,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Renders a 3-panel publication-ready visualization of retention budget economics:
          Panel 1: Net Saved Revenue vs Budget curve across scenarios.
          Panel 2: Customers Targeted and Expected Accounts Saved vs Spend.
          Panel 3: Cumulative Net Expected Value vs Customers Targeted (Retention Frontier).

        Args:
            comparison_df: Multi-scenario summary DataFrame.
            budget_sweep_df: Budget sensitivity DataFrame.
            target_records_df: Detailed segmented DataFrame from baseline run.
            output_path: Path to save PNG image.

        Returns:
            Saved Path.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.5))

        # Panel 1: Budget Sensitivity & Marginal Returns
        if not budget_sweep_df.empty:
            budgets = budget_sweep_df["budget_allocated"].values
            net_saved = budget_sweep_df["expected_net_saved_revenue"].values
            spend = budget_sweep_df["budget_spent"].values

            ax1.plot(budgets, net_saved, marker="o", color="#1B365D", linewidth=2.5, label="Expected Net Savings ($)")
            ax1.plot(budgets, spend, marker="s", color="#D9534F", linestyle="--", linewidth=2.0, label="Campaign Cost ($)")
            ax1.fill_between(budgets, 0, net_saved, color="#1B365D", alpha=0.1)

            # Mark maximum profit point
            max_idx = np.argmax(net_saved)
            max_b = budgets[max_idx]
            max_net = net_saved[max_idx]
            ax1.scatter([max_b], [max_net], color="#2ECC71", s=120, zorder=5)
            ax1.annotate(
                f"Peak Net: ${max_net:,.0f}\n(Budget: ${max_b:,.0f})",
                xy=(max_b, max_net),
                xytext=(max_b * 0.7, max_net * 0.85),
                arrowprops=dict(facecolor="#2ECC71", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold",
                color="#1B365D",
            )

        ax1.set_title("Budget Sensitivity & Diminishing Returns", fontsize=13, fontweight="bold", pad=12)
        ax1.set_xlabel("Campaign Budget ($)", fontsize=11)
        ax1.set_ylabel("Financial Value ($)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left")

        # Panel 2: Multi-Scenario Comparison (Bar Chart)
        if not comparison_df.empty:
            scenarios = comparison_df["scenario_name"].values
            x = np.arange(len(scenarios))
            width = 0.35

            net_rev = comparison_df["expected_net_saved_revenue"].values
            roi = comparison_df["expected_campaign_roi_pct"].values

            rects1 = ax2.bar(x - width / 2, net_rev, width, label="Net Saved Rev ($)", color="#2E86AB")
            ax2_twin = ax2.twinx()
            rects2 = ax2_twin.bar(x + width / 2, roi, width, label="Campaign ROI (%)", color="#F39C12", alpha=0.85)

            ax2.set_ylabel("Net Saved Revenue ($)", fontsize=11, color="#2E86AB")
            ax2_twin.set_ylabel("Campaign ROI (%)", fontsize=11, color="#F39C12")
            ax2.set_xticks(x)
            ax2.set_xticklabels(scenarios, rotation=15, ha="right", fontsize=10, fontweight="bold")
            ax2.set_title("What-If Scenario Comparison", fontsize=13, fontweight="bold", pad=12)
            ax2.grid(True, linestyle=":", alpha=0.4)

            # Values on bars
            for r in rects1:
                h = r.get_height()
                ax2.annotate(f"${h:,.0f}", xy=(r.get_x() + r.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

        # Panel 3: Cumulative Net Expected Value vs Customers Targeted (Retention Frontier)
        if "simulated_expected_net_value" in target_records_df.columns:
            sorted_ev = target_records_df["simulated_expected_net_value"].values
            cum_net = np.cumsum(sorted_ev)
            cust_count = np.arange(1, len(cum_net) + 1)

            # Random targeting baseline
            mean_ev = np.mean(sorted_ev)
            random_cum = cust_count * mean_ev

            ax3.plot(cust_count, cum_net, color="#27AE60", linewidth=2.5, label="Optimal EV Ranking")
            ax3.plot(cust_count, random_cum, color="#7F8C8D", linestyle=":", linewidth=2.0, label="Random Outreach")
            ax3.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)

            # Highlight positive EV cutoff
            pos_mask = sorted_ev > 0
            if np.any(pos_mask):
                pos_cutoff_count = int(np.sum(pos_mask))
                peak_cum = cum_net[pos_cutoff_count - 1]
                ax3.scatter([pos_cutoff_count], [peak_cum], color="#E74C3C", s=100, zorder=5)
                ax3.annotate(
                    f"Rational Stop: {pos_cutoff_count} accounts\n(EV <= 0 beyond this point)",
                    xy=(pos_cutoff_count, peak_cum),
                    xytext=(pos_cutoff_count * 0.45, peak_cum * 0.75),
                    arrowprops=dict(facecolor="#E74C3C", shrink=0.08, width=1.5, headwidth=6),
                    fontweight="bold",
                    color="#900C3F",
                )

        ax3.set_title("Customer Retention Frontier (Cumulative EV)", fontsize=13, fontweight="bold", pad=12)
        ax3.set_xlabel("Number of Customers Targeted", fontsize=11)
        ax3.set_ylabel("Cumulative Expected Net Value ($)", fontsize=11)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="lower right")

        plt.suptitle("Retention Budget Simulator & Financial Sensitivity Analysis", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved retention budget scenario plot to: {out_file}")
        return out_file
