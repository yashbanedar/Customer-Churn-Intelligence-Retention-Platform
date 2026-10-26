"""
Helper Functions and Analytical Utilities for the Streamlit Executive Dashboard.
Separates data processing, formatting, and mathematical simulation logic
from Streamlit UI rendering to ensure modularity and unit testability.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.business.retention_simulator import (
    CAUSAL_DISCLAIMER,
    CLVCalculationMode,
    RankingStrategy,
    RetentionAssumptionConfig,
    RetentionBudgetSimulator,
    SimulationOutcome,
)
from src.config import BASE_DIR, get_settings
from src.data.db_loader import get_db_engine


def format_currency(val: Union[int, float]) -> str:
    """Formats numeric values into standard USD currency strings."""
    if val is None or np.isnan(val):
        return "$0.00"
    return f"${val:,.2f}"


def format_percentage(val: Union[int, float], decimals: int = 1) -> str:
    """Formats decimal or percentage values into string with percent symbol."""
    if val is None or np.isnan(val):
        return "0.0%"
    return f"{val:.{decimals}f}%"


def compute_portfolio_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes executive KPIs from customer records.
    Supports either 'churn_label', 'churn', or 'Churn' columns,
    and 'monthly_charges' or 'MonthlyCharges'.
    """
    total = len(df)
    if total == 0:
        return {
            "total_customers": 0,
            "churned_customers": 0,
            "churn_rate_pct": 0.0,
            "total_mrr": 0.0,
            "mrr_at_risk": 0.0,
            "annualized_arr_at_risk": 0.0,
            "avg_revenue_per_user": 0.0,
        }

    # Identify churn column
    churn_col = None
    for cand in ["churn_label", "churn", "Churn"]:
        if cand in df.columns:
            churn_col = cand
            break

    if churn_col is not None:
        if df[churn_col].dtype == object or str(df[churn_col].iloc[0]).lower() in ("yes", "no"):
            churn_mask = df[churn_col].astype(str).str.lower() == "yes"
        else:
            churn_mask = df[churn_col].astype(int) == 1
        churned = int(churn_mask.sum())
    else:
        churn_mask = pd.Series([False] * total)
        churned = 0

    churn_rate = (churned / total * 100.0) if total > 0 else 0.0

    # Monthly charges
    m_col = "monthly_charges" if "monthly_charges" in df.columns else (
        "MonthlyCharges" if "MonthlyCharges" in df.columns else None
    )
    if m_col is not None:
        charges = pd.to_numeric(df[m_col], errors="coerce").fillna(0.0)
        total_mrr = float(charges.sum())
        mrr_at_risk = float(charges[churn_mask].sum())
        arpu = float(charges.mean())
    else:
        total_mrr = 0.0
        mrr_at_risk = 0.0
        arpu = 0.0

    return {
        "total_customers": total,
        "churned_customers": churned,
        "churn_rate_pct": round(churn_rate, 2),
        "total_mrr": round(total_mrr, 2),
        "mrr_at_risk": round(mrr_at_risk, 2),
        "annualized_arr_at_risk": round(mrr_at_risk * 12.0, 2),
        "avg_revenue_per_user": round(arpu, 2),
    }


def filter_customer_records(
    df: pd.DataFrame,
    risk_filter: str = "All",
    contract_filter: str = "All",
    internet_filter: str = "All",
    search_query: str = "",
) -> pd.DataFrame:
    """
    Filters customer DataFrame based on UI dropdown selections and search query.
    """
    filtered = df.copy()

    # Search query on customer ID
    id_col = "customer_id" if "customer_id" in filtered.columns else (
        "customerID" if "customerID" in filtered.columns else (
            "observed_customer_id" if "observed_customer_id" in filtered.columns else None
        )
    )
    if search_query and id_col in filtered.columns:
        filtered = filtered[
            filtered[id_col].astype(str).str.contains(search_query.strip(), case=False, na=False)
        ]

    # Risk tier filter
    risk_col = "predicted_risk_tier" if "predicted_risk_tier" in filtered.columns else (
        "risk_tier" if "risk_tier" in filtered.columns else None
    )
    if risk_filter != "All" and risk_col in filtered.columns:
        filtered = filtered[filtered[risk_col] == risk_filter]

    # Contract filter
    c_col = "contract" if "contract" in filtered.columns else (
        "Contract" if "Contract" in filtered.columns else (
            "observed_contract_type" if "observed_contract_type" in filtered.columns else None
        )
    )
    if contract_filter != "All" and c_col in filtered.columns:
        filtered = filtered[filtered[c_col] == contract_filter]

    # Internet Service filter
    i_col = "internet_service" if "internet_service" in filtered.columns else (
        "InternetService" if "InternetService" in filtered.columns else None
    )
    if internet_filter != "All" and i_col in filtered.columns:
        filtered = filtered[filtered[i_col] == internet_filter]

    return filtered


def run_interactive_retention_simulation(
    customers_df: pd.DataFrame,
    probabilities: np.ndarray,
    budget: float,
    cost_per_offer: float,
    assumed_save_rate: float,
    fixed_clv: float = 450.0,
    clv_mode_str: str = "fixed",
    require_positive_ev: bool = True,
) -> Tuple[SimulationOutcome, pd.DataFrame]:
    """
    Executes real-time retention budget simulation for dashboard interactive sliders.
    """
    clv_mode = CLVCalculationMode(clv_mode_str.lower())
    cfg = RetentionAssumptionConfig(
        scenario_name="Interactive Dashboard Simulation",
        budget=float(budget),
        cost_per_offer=float(cost_per_offer),
        assumed_offer_success_rate=float(assumed_save_rate),
        clv_mode=clv_mode,
        fixed_clv=float(fixed_clv),
        profit_margin=0.80,
        clv_horizon_months=12,
        ranking_strategy=RankingStrategy.EXPECTED_NET_VALUE,
        require_positive_expected_value=require_positive_ev,
    )
    simulator = RetentionBudgetSimulator(config=cfg)
    outcome, res_df = simulator.simulate(customers_df, probabilities, config=cfg)
    return outcome, res_df


def load_live_database_summary() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Safely probes PostgreSQL / SQLite database and returns connection status and table rows.
    """
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            row_count = conn.execute(text("SELECT COUNT(*) FROM customer_contracts")).scalar()
        return True, engine.dialect.name, {"total_rows": int(row_count)}
    except Exception as e:
        return False, "disconnected", {"error": str(e)}
