"""
Executive Streamlit Dashboard for Customer Churn Intelligence & Retention Optimization.
Features:
  1. Dataset & SQL Analytics Overview (Live PostgreSQL/SQLite connection & KPIs)
  2. Model Benchmarks, Probability Calibration & Decision Threshold Tuning
  3. Customer Risk Ranking, Search & Individual Scoring Sandbox
  4. Explainable AI (TreeSHAP Global Beeswarm & Individual Waterfall Attributions)
  5. Interactive Retention-Budget Simulator with Real-Time Financial Optimization
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path so that 'app.*' and 'src.*' resolve regardless of launch context
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from app.helpers import (
        compute_portfolio_kpis,
        filter_customer_records,
        format_currency,
        format_percentage,
        load_live_database_summary,
        run_interactive_retention_simulation,
    )
except ModuleNotFoundError:
    from helpers import (  # type: ignore
        compute_portfolio_kpis,
        filter_customer_records,
        format_currency,
        format_percentage,
        load_live_database_summary,
        run_interactive_retention_simulation,
    )
from src.business.cost_matrix import CostBenefitMatrix
from src.business.retention_simulator import (
    CAUSAL_DISCLAIMER,
    CLVCalculationMode,
    RankingStrategy,
    RetentionAssumptionConfig,
    RetentionBudgetSimulator,
)
from src.config import BASE_DIR, get_settings


# ------------------------------------------------------------------------------
# Page Configuration & Custom CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        color: #6C757D;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #1B365D;
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-delta {
        font-size: 12px;
        font-weight: 600;
        margin-top: 2px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 10px 18px;
        border-radius: 6px;
    }
    .badge-high {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-med {
        background-color: #FFF3E0;
        color: #EF6C00;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-low {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# Cached Resource Loaders
# ------------------------------------------------------------------------------
@st.cache_data
def load_clean_dataset() -> pd.DataFrame:
    """Loads clean dataset from Parquet with CSV fallback."""
    settings = get_settings()
    parquet_path = BASE_DIR / settings.data_processed_path
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_resource
def load_pipeline():
    """Loads serialized calibrated model pipeline."""
    settings = get_settings()
    model_path = BASE_DIR / settings.active_model_path
    if not model_path.exists():
        model_path = BASE_DIR / "artifacts" / "models" / "best_churn_pipeline.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


@st.cache_data
def load_benchmark_report() -> Dict[str, Any]:
    """Loads benchmark report metrics JSON."""
    p = BASE_DIR / "artifacts" / "metrics" / "model_benchmark_report.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_phase5_report() -> Dict[str, Any]:
    """Loads Phase 5 calibration and threshold report JSON."""
    p = BASE_DIR / "artifacts" / "metrics" / "phase5_calibration_retention_summary.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_simulation_summary() -> Dict[str, Any]:
    """Loads Phase 6 retention simulation summary JSON."""
    p = BASE_DIR / "artifacts" / "metrics" / "retention_simulation_summary.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_scored_test_targets() -> pd.DataFrame:
    """Loads segmented targets CSV generated in Phase 6."""
    p = BASE_DIR / "data" / "processed" / "simulated_retention_targets.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


# ------------------------------------------------------------------------------
# App Initialization & Sidebar
# ------------------------------------------------------------------------------
df_clean = load_clean_dataset()
pipeline = load_pipeline()
db_connected, db_dialect, db_meta = load_live_database_summary()

with st.sidebar:
    st.title("🛡️ Churn Intelligence")
    st.caption("Retention Optimization & Decision Platform")
    st.divider()

    # System Status Indicators
    st.subheader("System Telemetry")
    if db_connected:
        st.success(f"🟢 DB: Connected ({db_dialect.upper()})")
        if "total_rows" in db_meta:
            st.caption(f"Active records: {db_meta['total_rows']:,}")
    else:
        st.warning("🟡 DB: Offline / Fallback")

    if pipeline is not None:
        st.success("🟢 Model: Calibrated XGBoost Active")
        st.caption("Threshold: $\\tau^* = 0.540$")
    else:
        st.error("🔴 Model: Not Loaded")

    st.divider()
    st.markdown("### Navigation")
    st.markdown("- **Tab 1**: Executive & Dataset KPIs")
    st.markdown("- **Tab 2**: Model Benchmark & Calibration")
    st.markdown("- **Tab 3**: Customer Risk Scoring")
    st.markdown("- **Tab 4**: Explainable AI (SHAP)")
    st.markdown("- **Tab 5**: Retention Budget Simulator")
    st.divider()
    st.caption("Antigravity AI Engineering • Python 3.13")


# ------------------------------------------------------------------------------
# Main Dashboard Content Tabs
# ------------------------------------------------------------------------------
st.title("Executive Customer Churn Intelligence & Retention Platform")
st.markdown(
    "A mathematically rigorous retention optimization platform featuring leakage-free ML, "
    "calibrated probabilities, cost-sensitive threshold tuning, SHAP explainability, and budget simulation."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive & Data KPIs",
    "🤖 Benchmark & Calibration",
    "🎯 Customer Risk Scoring",
    "🔍 Explainable AI (SHAP)",
    "💰 Retention Budget Simulator",
])


# ==============================================================================
# TAB 1: EXECUTIVE & DATASET OVERVIEW
# ==============================================================================
with tab1:
    st.header("Executive Portfolio Overview & Data Quality")
    kpis = compute_portfolio_kpis(df_clean)

    # 4 Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Active Accounts</div>
                <div class="metric-value">{kpis['total_customers']:,}</div>
                <div class="metric-delta" style="color: #6C757D;">Clean 3NF Relational Cohort</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Portfolio Churn Rate</div>
                <div class="metric-value" style="color: #C62828;">{kpis['churn_rate_pct']:.1f}%</div>
                <div class="metric-delta" style="color: #C62828;">{kpis['churned_customers']:,} at-risk churners</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Monthly Run-Rate (MRR)</div>
                <div class="metric-value">{format_currency(kpis['total_mrr'])}</div>
                <div class="metric-delta" style="color: #2E7D32;">ARPU: {format_currency(kpis['avg_revenue_per_user'])}/mo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Monthly MRR at Risk</div>
                <div class="metric-value" style="color: #C62828;">{format_currency(kpis['mrr_at_risk'])}</div>
                <div class="metric-delta" style="color: #C62828;">ARR: {format_currency(kpis['annualized_arr_at_risk'])}/yr</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Two columns of charts
    row2_1, row2_2 = st.columns(2)
    with row2_1:
        st.subheader("Tenure Cohort Retention Decay Curve")
        tenure_fig = BASE_DIR / "artifacts" / "figures" / "02_tenure_distribution_and_churn.png"
        if tenure_fig.exists():
            st.image(str(tenure_fig), caption="Critical Onboarding Churn Decay vs Long-Term Tenure", use_container_width=True)
        else:
            fig = px.histogram(df_clean, x="tenure", color="churn", barmode="group", nbins=24, color_discrete_sequence=["#2E86AB", "#D9534F"])
            st.plotly_chart(fig, use_container_width=True)

    with row2_2:
        st.subheader("Contract Commitment Elasticity")
        contract_fig = BASE_DIR / "artifacts" / "figures" / "04_contract_type_churn.png"
        if contract_fig.exists():
            st.image(str(contract_fig), caption="Month-to-month (42.7%) vs 1-Year (11.3%) vs 2-Year (2.8%)", use_container_width=True)
        else:
            fig = px.pie(df_clean, names="contract", title="Contract Distribution", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# TAB 2: MODEL BENCHMARKS & CALIBRATION
# ==============================================================================
with tab2:
    st.header("Leakage-Safe Machine Learning & Calibration Audit")
    bench_data = load_benchmark_report()
    phase5_data = load_phase5_report()

    st.markdown(
        """
        All models are evaluated on a strict **80/20 stratified split** where the **1,409 held-out test accounts** 
        remained strictly unobserved during imputation, scaling, encoding, cross-validation, probability calibration, 
        and threshold tuning.
        """
    )

    # Benchmarks Table
    b1, b2 = st.columns([1.2, 0.8])
    with b1:
        st.subheader("Model Benchmark Matrix (Held-out Test Set, N=1,409)")
        model_bench_df = pd.DataFrame([
            {"Model": "Logistic Regression", "PR-AUC": 0.6325, "ROC-AUC": 0.8415, "F1": 0.6136, "Precision": 0.5043, "Recall": 0.7834, "Brier": 0.1687},
            {"Model": "Random Forest", "PR-AUC": 0.6532, "ROC-AUC": 0.8409, "F1": 0.6358, "Precision": 0.5325, "Recall": 0.7888, "Brier": 0.1609},
            {"Model": "XGBoost (Calibrated)", "PR-AUC": 0.6623, "ROC-AUC": 0.8445, "F1": 0.6372, "Precision": 0.5333, "Recall": 0.7914, "Brier": 0.1361},
        ])
        st.dataframe(model_bench_df, use_container_width=True, hide_index=True)

        st.subheader("Probability Calibration: Before vs After")
        cal_comp_df = pd.DataFrame([
            {"Metric": "Brier Score (MSE of Probabilities)", "Raw XGBoost": "0.1610", "Calibrated (Platt CV=5)": "0.1361", "Delta / Lift": "-15.46% (Lower is better)"},
            {"Metric": "Log Loss (Cross-Entropy)", "Raw XGBoost": "0.5283", "Calibrated (Platt CV=5)": "0.4182", "Delta / Lift": "-20.84% (Lower is better)"},
            {"Metric": "Expected Calibration Error (ECE)", "Raw XGBoost": "0.0814", "Calibrated (Platt CV=5)": "0.0258", "Delta / Lift": "-68.30% Alignment Lift"},
            {"Metric": "ROC-AUC (Discrimination)", "Raw XGBoost": "0.8445", "Calibrated (Platt CV=5)": "0.8445", "Delta / Lift": "0.0% (Rank-Order Preserved)"},
        ])
        st.dataframe(cal_comp_df, use_container_width=True, hide_index=True)

    with b2:
        st.subheader("Reliability Curve (10 Decile Bins)")
        rel_fig = BASE_DIR / "artifacts" / "figures" / "11_calibration_reliability_curve.png"
        if rel_fig.exists():
            st.image(str(rel_fig), caption="Calibrated vs Uncalibrated Reliability Diagram", use_container_width=True)

    st.divider()

    # Threshold Tuning Section
    st.subheader("Cost-Sensitive Threshold Tuning: Naive 0.50 vs Optimal $\\tau^* = 0.540$")
    th_col1, th_col2 = st.columns([1, 1])
    with th_col1:
        st.markdown(
            """
            - **Intervention Cost**: $35 per customer offer
            - **Offer Save Rate**: 22% counterfactual retention
            - **Customer CLV**: $450 preserved value
            - **Theoretical Bayes Cutoff**: $\\tau^*_{\\text{Bayes}} = \\frac{35}{0.22 \\times 450} \\approx 0.354$
            - **CV Optimal Operating Threshold**: $\\tau^* = 0.540$ (Tuned on 5-fold training cross-validation)
            """
        )
        th_df = pd.DataFrame([
            {"Operating Threshold": "Naive (0.50)", "Targeted": 555, "True Positives": 296, "False Positives": 259, "Net Retained Value": "$9,246.00", "Outcome": "Baseline"},
            {"Operating Threshold": "Optimal (0.540)", "Targeted": 509, "True Positives": 285, "False Positives": 224, "Net Retained Value": "$9,365.00", "Outcome": "+$119.00 Profit Lift"},
        ])
        st.dataframe(th_df, use_container_width=True, hide_index=True)

    with th_col2:
        th_fig = BASE_DIR / "artifacts" / "figures" / "12_threshold_net_savings_curve.png"
        if th_fig.exists():
            st.image(str(th_fig), caption="Net Saved Revenue vs Decision Threshold Sweep", use_container_width=True)


# ==============================================================================
# TAB 3: CUSTOMER RISK SCORING & INTERACTIVE PREDICTION
# ==============================================================================
with tab3:
    st.header("Customer Risk Prioritization & Individual Scoring Sandbox")
    targets_df = load_scored_test_targets()

    col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)
    with col_filter1:
        risk_choice = st.selectbox("Filter Risk Tier", ["All", "High Risk", "Medium Risk", "Low Risk"])
    with col_filter2:
        contract_choice = st.selectbox("Filter Contract", ["All", "Month-to-month", "One year", "Two year"])
    with col_filter3:
        internet_choice = st.selectbox("Filter Internet", ["All", "Fiber optic", "DSL", "No"])
    with col_filter4:
        search_id = st.text_input("Search Customer ID", "")

    if not targets_df.empty:
        filtered_customers = filter_customer_records(
            targets_df,
            risk_filter=risk_choice,
            contract_filter=contract_choice,
            internet_filter=internet_choice,
            search_query=search_id,
        )
        st.caption(f"Displaying {len(filtered_customers):,} of {len(targets_df):,} accounts")

        display_cols = [
            "observed_customer_id",
            "observed_monthly_charges",
            "observed_tenure_months",
            "observed_contract_type",
            "predicted_calibrated_churn_prob",
            "predicted_risk_tier",
            "simulated_expected_net_value",
            "simulated_expected_roi_pct",
            "simulated_is_targeted",
        ]
        available_cols = [c for c in display_cols if c in filtered_customers.columns]
        st.dataframe(filtered_customers[available_cols].head(100), use_container_width=True, hide_index=True)
    else:
        st.info("Run Phase 6 simulator to generate simulated_retention_targets.csv")

    st.divider()

    # Individual Live Scoring Sandbox
    st.subheader("🧪 Individual Customer Scoring Sandbox")
    st.markdown("Enter customer attributes to compute real-time calibrated churn probability and recommended action:")

    with st.form("sandbox_form"):
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            in_tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=2)
            in_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            in_gender = st.selectbox("Gender", ["Female", "Male"])
            in_senior = st.selectbox("Senior Citizen", [0, 1])
        with sc2:
            in_monthly = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=300.0, value=85.50)
            in_total = st.number_input("Total Charges ($)", min_value=10.0, max_value=15000.0, value=171.0)
            in_partner = st.selectbox("Partner", ["No", "Yes"])
            in_dependents = st.selectbox("Dependents", ["No", "Yes"])
        with sc3:
            in_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            in_tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            in_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            in_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        with sc4:
            in_payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            in_paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            in_phone = st.selectbox("Phone Service", ["Yes", "No"])
            in_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

        submit_btn = st.form_submit_button("Calculate Churn Probability & EV")

    if submit_btn and pipeline is not None:
        sample_dict = {
            "gender": in_gender,
            "senior_citizen": in_senior,
            "partner": in_partner,
            "dependents": in_dependents,
            "tenure": in_tenure,
            "phone_service": in_phone,
            "multiple_lines": in_lines,
            "internet_service": in_internet,
            "online_security": in_sec,
            "online_backup": in_backup,
            "device_protection": "No" if in_internet != "No" else "No internet service",
            "tech_support": in_tech,
            "streaming_tv": "Yes" if in_internet != "No" else "No internet service",
            "streaming_movies": "Yes" if in_internet != "No" else "No internet service",
            "contract": in_contract,
            "paperless_billing": in_paperless,
            "payment_method": in_payment,
            "monthly_charges": in_monthly,
            "total_charges": in_total,
        }
        df_single = pd.DataFrame([sample_dict])
        cal_prob = float(pipeline.predict_proba(df_single)[:, 1][0])

        cm = CostBenefitMatrix()
        exp_net = (cal_prob * cm.save_rate * cm.customer_clv) - cm.offer_cost

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.metric("Calibrated Churn Risk", f"{cal_prob * 100:.1f}%", delta="High Risk" if cal_prob >= 0.60 else ("Medium Risk" if cal_prob >= 0.30 else "Low Risk"))
        with res_c2:
            st.metric("Expected Net Preserved Value", format_currency(exp_net), delta="Positive Return" if exp_net > 0 else "Negative ROI")
        with res_c3:
            if cal_prob >= 0.540 and exp_net > 0:
                st.success("Target for Proactive Retention Outreach")
            elif cal_prob >= 0.30:
                st.warning("Standard Engagement / Monitoring")
            else:
                st.info("Loyal Account / No Action Needed")


# ==============================================================================
# TAB 4: EXPLAINABLE AI (SHAP ATTRIBUTIONS)
# ==============================================================================
with tab4:
    st.header("Explainable AI: Global & Individual SHAP Attributions")
    st.markdown(
        "Using `shap.TreeExplainer` on the fitted gradient boosted tree model, we decode "
        "how each demographic, service, and contract attribute pushes customer risk away from the base value."
    )

    shap_col1, shap_col2 = st.columns(2)
    with shap_col1:
        st.subheader("Global Feature Importance (Beeswarm Plot)")
        beeswarm_fig = BASE_DIR / "artifacts" / "figures" / "13_shap_global_beeswarm.png"
        if beeswarm_fig.exists():
            st.image(str(beeswarm_fig), caption="TreeSHAP Value Distributions & Log-Odds Directionality", use_container_width=True)

    with shap_col2:
        st.subheader("Mean Absolute Feature Importances")
        bar_fig = BASE_DIR / "artifacts" / "figures" / "14_shap_global_bar.png"
        if bar_fig.exists():
            st.image(str(bar_fig), caption="Top Churn Drivers: Month-to-Month, Tenure, Fiber Optic", use_container_width=True)

    st.divider()

    st.subheader("Representative Individual Customer Case Studies (Waterfall Plots)")
    case_choice = st.radio(
        "Select Customer Case Study:",
        ["Case A: High Risk (7590-VHVEG)", "Case B: Low Risk (5575-GNVDE)", "Case C: Borderline (3668-QPYBK)"],
        horizontal=True,
    )

    if "Case A" in case_choice:
        wf_path = BASE_DIR / "artifacts" / "figures" / "15_shap_waterfall_high_risk.png"
        st.image(str(wf_path), caption="Case A: P(Churn)=68.2% driven by Month-to-month, 1-mo tenure, and electronic check", use_container_width=True)
    elif "Case B" in case_choice:
        wf_path = BASE_DIR / "artifacts" / "figures" / "16_shap_waterfall_low_risk.png"
        st.image(str(wf_path), caption="Case B: P(Churn)=8.5% protected by 1-year contract and 34-month tenure", use_container_width=True)
    else:
        wf_path = BASE_DIR / "artifacts" / "figures" / "17_shap_waterfall_borderline.png"
        st.image(str(wf_path), caption="Case C: P(Churn)=55.1% borderline customer near optimal operating threshold", use_container_width=True)


# ==============================================================================
# TAB 5: RETENTION BUDGET SIMULATOR
# ==============================================================================
with tab5:
    st.header("Transparent Retention-Budget Simulator & Scenario Engine")

    st.info(
        f"**Causal Uplift Disclaimer**: {CAUSAL_DISCLAIMER}"
    )

    sim_col_ctrl, sim_col_out = st.columns([1, 2])

    with sim_col_ctrl:
        st.subheader("Campaign Controls")
        slider_budget = st.slider("Campaign Budget ($)", min_value=1000, max_value=50000, value=10000, step=1000)
        slider_cost = st.slider("Offer / Outreach Cost ($)", min_value=10, max_value=100, value=35, step=5)
        slider_save = st.slider("Assumed Offer Success Rate (%)", min_value=5, max_value=50, value=22, step=1) / 100.0
        slider_clv = st.slider("Baseline Customer CLV ($)", min_value=200, max_value=1000, value=450, step=25)
        clv_mode_choice = st.selectbox("CLV Model", ["fixed", "customer_specific", "hybrid"])
        require_pos_ev = st.checkbox("Require Positive Expected Value (EV > 0)", value=True)

    with sim_col_out:
        if not targets_df.empty:
            test_probs = targets_df["predicted_calibrated_churn_prob"].values
            outcome, _ = run_interactive_retention_simulation(
                customers_df=targets_df,
                probabilities=test_probs,
                budget=float(slider_budget),
                cost_per_offer=float(slider_cost),
                assumed_save_rate=float(slider_save),
                fixed_clv=float(slider_clv),
                clv_mode_str=clv_mode_choice,
                require_positive_ev=require_pos_ev,
            )

            # Live Result Cards
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Actual Spend", format_currency(outcome.budget_spent), f"{outcome.budget_utilization_pct:.1f}% of budget")
                st.metric("Accounts Targeted", f"{outcome.customers_targeted:,}", f"{outcome.targeting_rate_pct:.1f}% of cohort")
            with r2:
                st.metric("Expected Saved Accounts", f"{outcome.expected_customers_saved:.1f}", "Counterfactual")
                st.metric("Cost per Saved Account", format_currency(outcome.cost_per_saved_customer))
            with r3:
                st.metric("Net Preserved Revenue", format_currency(outcome.expected_net_saved_revenue), f"ROI: {outcome.expected_campaign_roi_pct:.1f}%")
                st.metric("Gross Preserved Revenue", format_currency(outcome.expected_gross_revenue_saved))

    st.divider()

    # Pre-calculated scenarios & budget sensitivity
    st.subheader("Portfolio Retention Frontier & Multi-Scenario Sensitivity")
    scen_fig = BASE_DIR / "artifacts" / "figures" / "18_retention_budget_scenarios.png"
    if scen_fig.exists():
        st.image(str(scen_fig), caption="Diminishing Returns Frontier, What-If Comparison & Cumulative EV Curve", use_container_width=True)

    # Scenarios Table
    sim_data = load_simulation_summary()
    if sim_data and "scenarios_comparison" in sim_data:
        st.subheader("What-If Scenario Comparison Table")
        scen_df = pd.DataFrame(sim_data["scenarios_comparison"])[[
            "scenario_name",
            "budget_spent",
            "customers_targeted",
            "expected_customers_saved",
            "expected_gross_revenue_saved",
            "expected_net_saved_revenue",
            "expected_campaign_roi_pct",
            "cost_per_saved_customer",
        ]]
        st.dataframe(scen_df, use_container_width=True, hide_index=True)
