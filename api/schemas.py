"""
Pydantic v2 Schemas for FastAPI Prediction and Analytics Microservice.
Provides strict type-checking, field constraints, categorical domain enforcement,
and Swagger/OpenAPI documentation models.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class CustomerInput(BaseModel):
    """
    Validated demographic, service, and billing attributes for a single customer.
    Supports snake_case or PascalCase input keys.
    """
    customer_id: str = Field(
        default="CUST-PREDICT",
        description="Unique customer identifier",
        examples=["7590-VHVEG"],
    )
    gender: Literal["Male", "Female"] = Field(
        ...,
        description="Customer gender",
        examples=["Female"],
    )
    senior_citizen: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether customer is a senior citizen (0 for No, 1 for Yes)",
        examples=[0],
    )
    partner: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has a partner",
        examples=["Yes"],
    )
    dependents: Literal["Yes", "No"] = Field(
        ...,
        description="Whether the customer has dependents",
        examples=["No"],
    )
    tenure: int = Field(
        ...,
        ge=0,
        le=120,
        description="Number of months customer has stayed with the company",
        examples=[1],
    )
    phone_service: Literal["Yes", "No"] = Field(
        ...,
        description="Whether customer has telephone service",
        examples=["No"],
    )
    multiple_lines: Literal["No phone service", "No", "Yes"] = Field(
        ...,
        description="Whether customer has multiple phone lines",
        examples=["No phone service"],
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ...,
        description="Type of customer internet service connection",
        examples=["DSL"],
    )
    online_security: Literal["No internet service", "No", "Yes"] = Field(
        ...,
        description="Whether customer has online security add-on",
        examples=["No"],
    )
    online_backup: Literal["No internet service", "No", "Yes"] = Field(
        ...,
        description="Whether customer has online backup add-on",
        examples=["Yes"],
    )
    device_protection: Literal["No internet service", "No", "Yes"] = Field(
        ...,
        description="Whether customer has device protection plan",
        examples=["No"],
    )
    tech_support: Literal["No internet service", "No", "Yes"] = Field(
        ...,
        description="Whether customer has technical support add-on",
        examples=["No"],
    )
    streaming_tv: Literal["No internet service", "No", "Yes"] = Field(
        ...,
        description="Whether customer has streaming TV subscription",
        examples=["No"],
    )
    streaming_movies: Literal["No internet service", "No", "Yes"] = Field(
        ...,
        description="Whether customer has streaming movies subscription",
        examples=["No"],
    )
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ...,
        description="Customer commitment contract duration",
        examples=["Month-to-month"],
    )
    paperless_billing: Literal["Yes", "No"] = Field(
        ...,
        description="Whether customer uses paperless electronic billing",
        examples=["Yes"],
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ...,
        description="Payment billing channel",
        examples=["Electronic check"],
    )
    monthly_charges: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Current monthly charge amount in USD",
        examples=[29.85],
    )
    total_charges: float = Field(
        ...,
        ge=0.0,
        le=50000.0,
        description="Cumulative charges over customer lifecycle in USD",
        examples=[29.85],
    )

    @field_validator("senior_citizen", mode="before")
    @classmethod
    def parse_senior_citizen(cls, v: Any) -> int:
        if isinstance(v, bool):
            return 1 if v else 0
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean in ("1", "yes", "true"):
                return 1
            if v_clean in ("0", "no", "false"):
                return 0
        return int(v)


class PredictionOutput(BaseModel):
    """
    Inference response schema for a single customer risk assessment.
    """
    customer_id: str
    calibrated_churn_probability: float = Field(..., description="Post-calibration probability P(Churn=1)")
    predicted_churn: bool = Field(..., description="Binary classification at optimal decision threshold")
    decision_threshold: float = Field(..., description="Cost-sensitive decision threshold used (default 0.540)")
    risk_tier: Literal["High Risk", "Medium Risk", "Low Risk"]
    expected_gross_saved_value: float = Field(..., description="Expected preserved value: p * save_rate * CLV")
    expected_net_value: float = Field(..., description="Net financial value after offer cost")
    recommended_action: Literal[
        "Proactive Retention Outreach",
        "Standard Engagement",
        "Loyal / Monitor",
    ]
    key_drivers: List[str] = Field(
        default_factory=list,
        description="Primary qualitative risk factors identified for this profile",
    )


class BatchPredictionRequest(BaseModel):
    """
    Batch scoring request supporting up to 500 accounts simultaneously.
    """
    customers: List[CustomerInput] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of customer records to score",
    )
    custom_threshold: Optional[float] = Field(
        default=0.540,
        ge=0.05,
        le=0.95,
        description="Optional custom decision threshold override",
    )
    offer_cost: Optional[float] = Field(
        default=35.0,
        gt=0.0,
        description="Direct cost per retention offer ($)",
    )
    assumed_save_rate: Optional[float] = Field(
        default=0.22,
        ge=0.0,
        le=1.0,
        description="Assumed offer counterfactual retention rate",
    )
    customer_clv: Optional[float] = Field(
        default=450.0,
        gt=0.0,
        description="Customer lifetime value preserved if retained ($)",
    )


class BatchPredictionSummary(BaseModel):
    """
    Statistical aggregates across a scored batch of customers.
    """
    total_customers: int
    mean_calibrated_probability: float
    high_risk_count: int
    targeted_for_outreach_count: int
    targeting_rate_pct: float
    total_expected_net_savings: float


class BatchPredictionResponse(BaseModel):
    """
    Complete batch scoring response containing individual scores and aggregate summary.
    """
    predictions: List[PredictionOutput]
    summary: BatchPredictionSummary
    causal_disclaimer: str = Field(
        default=(
            "Retention uplift and expected savings figures are based on user assumptions. "
            "Observational churn risk is not proof of causal retention response."
        )
    )


class HealthResponse(BaseModel):
    """
    Health check payload verifying microservice, ML model, and database readiness.
    """
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    model_type: str
    active_threshold: float
    database_connected: bool
    database_dialect: str
    database_tables: List[str]
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AnalyticsSummaryResponse(BaseModel):
    """
    Real-time database analytics snapshot loaded from PostgreSQL / SQLite.
    """
    total_customers: int
    churned_customers: int
    churn_rate_pct: float
    total_mrr: float
    mrr_at_risk: float
    avg_revenue_per_user: float
    contracts_breakdown: List[Dict[str, Any]]
    tenure_cohorts: List[Dict[str, Any]]
    database_dialect: str
