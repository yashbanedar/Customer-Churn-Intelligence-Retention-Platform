"""
FastAPI Microservice for Customer Churn Intelligence & Retention Optimization.
Provides validated single & batch inference, calibrated probability estimation,
cost-sensitive threshold evaluation, health checks, and database analytics.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import inspect, text

try:
    from api.schemas import (
        AnalyticsSummaryResponse,
        BatchPredictionRequest,
        BatchPredictionResponse,
        BatchPredictionSummary,
        CustomerInput,
        HealthResponse,
        PredictionOutput,
    )
except ModuleNotFoundError:
    from schemas import (  # type: ignore
        AnalyticsSummaryResponse,
        BatchPredictionRequest,
        BatchPredictionResponse,
        BatchPredictionSummary,
        CustomerInput,
        HealthResponse,
        PredictionOutput,
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
from src.data.db_loader import get_db_engine
from src.models.pipeline import ALL_FEATURE_COLUMNS


logger = logging.getLogger("ChurnAPI")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# Global in-memory model cache
ml_artifacts: Dict[str, Any] = {
    "pipeline": None,
    "model_path": None,
    "optimal_threshold": 0.540,
    "cost_matrix": None,
    "db_engine": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: loads ML pipeline and verifies database connectivity on startup.
    """
    settings = get_settings()
    logger.info("Initializing Churn Prediction Microservice...")

    # 1. Load ML Pipeline
    model_path = BASE_DIR / settings.active_model_path
    if not model_path.exists():
        fallback_path = BASE_DIR / "artifacts" / "models" / "best_churn_pipeline.joblib"
        if fallback_path.exists():
            model_path = fallback_path

    if model_path.exists():
        try:
            logger.info(f"Loading active model pipeline from: {model_path}")
            ml_artifacts["pipeline"] = joblib.load(model_path)
            ml_artifacts["model_path"] = str(model_path)
            logger.info("Model pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load pipeline from {model_path}: {e}")
            ml_artifacts["pipeline"] = None
    else:
        logger.warning(f"No serialized model found at {model_path}. Predict endpoints will fail until trained.")

    # 2. Setup Cost Matrix & Default Threshold
    ml_artifacts["optimal_threshold"] = 0.540
    ml_artifacts["cost_matrix"] = CostBenefitMatrix(
        offer_cost=settings.default_offer_cost,
        save_rate=settings.default_save_rate,
        customer_clv=450.0,
        horizon_months=settings.default_horizon_months,
    )

    # 3. Setup Database Engine
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        ml_artifacts["db_engine"] = engine
        logger.info(f"Database connection verified ({engine.dialect.name}).")
    except Exception as e:
        logger.warning(f"Database connection unavailable at startup: {e}")
        ml_artifacts["db_engine"] = None

    yield

    # Teardown
    logger.info("Shutting down Churn Prediction Microservice...")
    if ml_artifacts.get("db_engine"):
        ml_artifacts["db_engine"].dispose()


app = FastAPI(
    title="Customer Churn Intelligence & Retention Optimization API",
    version="1.0.0",
    description=(
        "Production-grade microservice for predictive churn scoring, probability calibration, "
        "cost-sensitive decision threshold optimization, and financial retention budget simulation."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for dashboard and frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Custom Exception Handlers
# ------------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Transforms Pydantic validation errors into structured, user-friendly JSON payloads.
    """
    error_details = []
    for err in exc.errors():
        loc = " -> ".join([str(x) for x in err.get("loc", []) if x != "body"])
        error_details.append({
            "field": loc,
            "message": err.get("msg"),
            "type": err.get("type"),
            "input_provided": err.get("input"),
        })

    logger.warning(f"Validation error on {request.method} {request.url.path}: {error_details}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "One or more input fields failed validation constraints.",
            "details": error_details,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Standardizes HTTP error responses.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches unhandled runtime exceptions with sanitized message output.
    """
    logger.exception(f"Unhandled error processing {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected server error occurred during request processing.",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def _format_customer_dataframe(customers: List[CustomerInput]) -> pd.DataFrame:
    """
    Converts a list of Pydantic CustomerInput models into a pandas DataFrame
    matching the exact column schema expected by the scikit-learn Pipeline.
    """
    records = []
    for c in customers:
        d = c.model_dump()
        records.append(d)
    df = pd.DataFrame(records)
    return df


def _extract_heuristic_key_drivers(row: pd.Series, prob: float) -> List[str]:
    """
    Extracts the most influential qualitative risk drivers for frontline staff
    based on empirical feature importances (Phase 3 & Phase 5 SHAP).
    """
    drivers = []
    if row.get("contract") == "Month-to-month":
        drivers.append("Month-to-month contract (High churn hazard)")
    if row.get("tenure", 0) <= 6:
        drivers.append(f"Short tenure ({row.get('tenure')} months, onboarding risk)")
    if row.get("internet_service") == "Fiber optic":
        drivers.append("Fiber optic subscription (High dissatisfaction/price elasticity)")
    if row.get("tech_support") == "No":
        drivers.append("Absence of Tech Support protection")
    if row.get("payment_method") == "Electronic check":
        drivers.append("Payment via Electronic check (High billing friction)")
    if row.get("monthly_charges", 0) >= 80.0:
        drivers.append(f"High monthly charges (${row.get('monthly_charges'):.2f}/mo)")

    if not drivers:
        if prob < 0.30:
            drivers.append("Long-term contract commitment & multi-service attachment (Loyal)")
        else:
            drivers.append("Moderate composite service friction")

    return drivers[:3]


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/", tags=["General"])
async def root():
    """Service metadata and links to interactive OpenAPI documentation."""
    return {
        "service": "Customer Churn Intelligence & Retention Optimization API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "analytics": "/analytics/summary",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint verifying model readiness, active decision threshold,
    and database connectivity.
    """
    pipeline_loaded = ml_artifacts["pipeline"] is not None
    db_connected = False
    dialect_name = "unknown"
    table_list: List[str] = []

    engine = ml_artifacts.get("db_engine")
    if engine is None:
        try:
            engine = get_db_engine()
            ml_artifacts["db_engine"] = engine
        except Exception:
            engine = None

    if engine is not None:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_connected = True
            dialect_name = engine.dialect.name
            insp = inspect(engine)
            table_list = insp.get_table_names()
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            db_connected = False

    overall_status = "healthy" if (pipeline_loaded and db_connected) else (
        "degraded" if pipeline_loaded else "unhealthy"
    )

    return HealthResponse(
        status=overall_status,
        model_loaded=pipeline_loaded,
        model_type="CalibratedClassifierCV(XGBoost)" if pipeline_loaded else "None",
        active_threshold=ml_artifacts["optimal_threshold"],
        database_connected=db_connected,
        database_dialect=dialect_name,
        database_tables=table_list,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
async def predict_single_customer(customer: CustomerInput):
    """
    Generates calibrated churn probability, optimal threshold classification,
    and net financial expected value for a single customer account.
    """
    pipeline = ml_artifacts.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded. Ensure artifacts are generated.",
        )

    df_single = _format_customer_dataframe([customer])
    try:
        prob = float(pipeline.predict_proba(df_single)[:, 1][0])
    except Exception as e:
        logger.error(f"Prediction pipeline error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution failed: {str(e)}",
        )

    tau = ml_artifacts["optimal_threshold"]
    is_churn_pred = bool(prob >= tau)

    # Risk Tier
    if prob >= 0.60:
        risk_tier = "High Risk"
    elif prob >= 0.30:
        risk_tier = "Medium Risk"
    else:
        risk_tier = "Low Risk"

    # Expected Value calculation
    cm: CostBenefitMatrix = ml_artifacts["cost_matrix"] or CostBenefitMatrix()
    exp_gross = prob * cm.save_rate * cm.customer_clv
    exp_net = exp_gross - cm.offer_cost

    # Recommended action based on optimal threshold and positive EV
    if is_churn_pred and exp_net > 0:
        recommendation = "Proactive Retention Outreach"
    elif prob >= 0.30:
        recommendation = "Standard Engagement"
    else:
        recommendation = "Loyal / Monitor"

    key_drivers = _extract_heuristic_key_drivers(df_single.iloc[0], prob)

    return PredictionOutput(
        customer_id=customer.customer_id,
        calibrated_churn_probability=round(prob, 4),
        predicted_churn=is_churn_pred,
        decision_threshold=tau,
        risk_tier=risk_tier,
        expected_gross_saved_value=round(exp_gross, 2),
        expected_net_value=round(exp_net, 2),
        recommended_action=recommendation,
        key_drivers=key_drivers,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch_customers(batch_request: BatchPredictionRequest):
    """
    High-throughput batch inference endpoint supporting up to 500 accounts simultaneously.
    Computes vector-accelerated predictions and portfolio summary metrics.
    """
    pipeline = ml_artifacts.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded.",
        )

    customers = batch_request.customers
    n_customers = len(customers)
    df_batch = _format_customer_dataframe(customers)

    try:
        probs = pipeline.predict_proba(df_batch)[:, 1]
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )

    tau = batch_request.custom_threshold or ml_artifacts["optimal_threshold"]
    c_cost = batch_request.offer_cost or 35.0
    r_save = batch_request.assumed_save_rate or 0.22
    clv_val = batch_request.customer_clv or 450.0

    predictions: List[PredictionOutput] = []
    targeted_count = 0
    high_risk_count = 0
    total_net_savings = 0.0

    for idx, prob in enumerate(probs):
        p = float(prob)
        cust = customers[idx]
        is_churn = bool(p >= tau)

        if p >= 0.60:
            risk_tier = "High Risk"
            high_risk_count += 1
        elif p >= 0.30:
            risk_tier = "Medium Risk"
        else:
            risk_tier = "Low Risk"

        exp_gross = p * r_save * clv_val
        exp_net = exp_gross - c_cost

        if is_churn and exp_net > 0:
            recommendation = "Proactive Retention Outreach"
            targeted_count += 1
            total_net_savings += exp_net
        elif p >= 0.30:
            recommendation = "Standard Engagement"
        else:
            recommendation = "Loyal / Monitor"

        row = df_batch.iloc[idx]
        key_drivers = _extract_heuristic_key_drivers(row, p)

        predictions.append(
            PredictionOutput(
                customer_id=cust.customer_id,
                calibrated_churn_probability=round(p, 4),
                predicted_churn=is_churn,
                decision_threshold=tau,
                risk_tier=risk_tier,
                expected_gross_saved_value=round(exp_gross, 2),
                expected_net_value=round(exp_net, 2),
                recommended_action=recommendation,
                key_drivers=key_drivers,
            )
        )

    targeting_rate_pct = (targeted_count / n_customers * 100.0) if n_customers > 0 else 0.0
    summary = BatchPredictionSummary(
        total_customers=n_customers,
        mean_calibrated_probability=round(float(np.mean(probs)), 4),
        high_risk_count=high_risk_count,
        targeted_for_outreach_count=targeted_count,
        targeting_rate_pct=round(targeting_rate_pct, 2),
        total_expected_net_savings=round(total_net_savings, 2),
    )

    return BatchPredictionResponse(
        predictions=predictions,
        summary=summary,
        causal_disclaimer=CAUSAL_DISCLAIMER,
    )


@app.get("/analytics/summary", response_model=AnalyticsSummaryResponse, tags=["Analytics"])
async def get_analytics_summary():
    """
    Executes live SQL analytical queries against PostgreSQL (or SQLite fallback)
    to return portfolio KPIs, MRR at risk, and contract churn breakdowns.
    """
    engine = ml_artifacts.get("db_engine")
    if engine is None:
        try:
            engine = get_db_engine()
            ml_artifacts["db_engine"] = engine
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Database engine is unavailable: {e}",
            )

    try:
        with engine.connect() as conn:
            # 1. Overall KPIs
            kpi_query = text("""
                SELECT 
                    COUNT(*) AS total_customers,
                    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
                    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
                    ROUND(SUM(monthly_charges), 2) AS total_mrr,
                    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END), 2) AS mrr_at_risk,
                    ROUND(AVG(monthly_charges), 2) AS avg_revenue_per_user
                FROM customer_contracts
            """)
            kpi_res = conn.execute(kpi_query).fetchone()

            # 2. Contracts breakdown
            contracts_query = text("""
                SELECT 
                    contract,
                    COUNT(*) AS total_customers,
                    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
                    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
                    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charge,
                    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END), 2) AS mrr_at_risk
                FROM customer_contracts
                GROUP BY contract
                ORDER BY churn_rate_pct DESC
            """)
            contracts_rows = [dict(row._mapping) for row in conn.execute(contracts_query).fetchall()]

            # 3. Tenure cohorts
            tenure_query = text("""
                SELECT 
                    CASE 
                        WHEN tenure <= 6 THEN '0 - 6 Mo (Onboarding)'
                        WHEN tenure <= 12 THEN '7 - 12 Mo (Cliff)'
                        WHEN tenure <= 24 THEN '13 - 24 Mo (Established)'
                        WHEN tenure <= 48 THEN '25 - 48 Mo (Mature)'
                        ELSE '49+ Mo (Loyal)'
                    END AS cohort,
                    COUNT(*) AS customer_count,
                    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_count,
                    ROUND(100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct,
                    ROUND(AVG(monthly_charges), 2) AS avg_monthly_spend
                FROM customer_contracts
                GROUP BY 1
                ORDER BY 1
            """)
            tenure_rows = [dict(row._mapping) for row in conn.execute(tenure_query).fetchall()]

        return AnalyticsSummaryResponse(
            total_customers=int(kpi_res[0]),
            churned_customers=int(kpi_res[1]),
            churn_rate_pct=float(kpi_res[2]),
            total_mrr=float(kpi_res[3]),
            mrr_at_risk=float(kpi_res[4]),
            avg_revenue_per_user=float(kpi_res[5]),
            contracts_breakdown=contracts_rows,
            tenure_cohorts=tenure_rows,
            database_dialect=engine.dialect.name,
        )
    except Exception as e:
        logger.error(f"Database query failure in /analytics/summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query execution failed: {str(e)}",
        )
