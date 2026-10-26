"""
Schema definitions and Pandera contracts for IBM Telco Customer Churn data.
Provides raw and clean dataset specifications, categorical domain sets, and column mappings.
"""

from typing import Dict, List, Set
import os
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

try:
    import pandera.pandas as pa
except (ImportError, AttributeError):
    import pandera as pa

from pandera import Check, Column, DataFrameSchema


# List of canonical 21 columns in raw IBM dataset
RAW_COLUMNS: List[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

# Standardized snake_case column names mapping
RAW_TO_SNAKE_CASE: Dict[str, str] = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

# Reverse mapping: snake_case to raw
SNAKE_CASE_TO_RAW: Dict[str, str] = {v: k for k, v in RAW_TO_SNAKE_CASE.items()}

# Domain definitions for categorical features
CATEGORICAL_DOMAINS: Dict[str, Set[str]] = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"No", "Yes", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"No", "Yes", "No internet service"},
    "OnlineBackup": {"No", "Yes", "No internet service"},
    "DeviceProtection": {"No", "Yes", "No internet service"},
    "TechSupport": {"No", "Yes", "No internet service"},
    "StreamingTV": {"No", "Yes", "No internet service"},
    "StreamingMovies": {"No", "Yes", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}

CLEAN_CATEGORICAL_DOMAINS: Dict[str, Set[str]] = {
    RAW_TO_SNAKE_CASE[k]: v for k, v in CATEGORICAL_DOMAINS.items()
}


# ==============================================================================
# Pandera Raw Schema Specification
# Note: TotalCharges is string/object in raw CSV due to 11 whitespace entries
# ==============================================================================
RAW_DATA_SCHEMA = DataFrameSchema(
    columns={
        "customerID": Column(
            pa.String,
            checks=[
                Check.str_length(min_value=5, max_value=20),
            ],
            nullable=False,
            unique=True,
            description="Unique customer ID",
        ),
        "gender": Column(
            pa.String,
            checks=Check.isin(["Male", "Female"]),
            nullable=False,
        ),
        "SeniorCitizen": Column(
            pa.Int,
            checks=Check.isin([0, 1]),
            nullable=False,
        ),
        "Partner": Column(
            pa.String,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "Dependents": Column(
            pa.String,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "tenure": Column(
            pa.Int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "PhoneService": Column(
            pa.String,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "MultipleLines": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No phone service"]),
            nullable=False,
        ),
        "InternetService": Column(
            pa.String,
            checks=Check.isin(["DSL", "Fiber optic", "No"]),
            nullable=False,
        ),
        "OnlineSecurity": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "OnlineBackup": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "DeviceProtection": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "TechSupport": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "StreamingTV": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "StreamingMovies": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "Contract": Column(
            pa.String,
            checks=Check.isin(["Month-to-month", "One year", "Two year"]),
            nullable=False,
        ),
        "PaperlessBilling": Column(
            pa.String,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "PaymentMethod": Column(
            pa.String,
            checks=Check.isin([
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]),
            nullable=False,
        ),
        "MonthlyCharges": Column(
            pa.Float,
            checks=Check.greater_than_or_equal_to(0.0),
            nullable=False,
        ),
        "TotalCharges": Column(
            pa.Object,
            nullable=False,
            description="Raw string TotalCharges containing spaces for 0 tenure",
        ),
        "Churn": Column(
            pa.String,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
    },
    coerce=False,
    strict="filter",
)


# ==============================================================================
# Pandera Clean Data Schema Specification
# TotalCharges converted to Float, churn_label integer indicator added
# ==============================================================================
CLEAN_DATA_SCHEMA = DataFrameSchema(
    columns={
        "customer_id": Column(
            pa.String,
            checks=[Check.str_length(min_value=5, max_value=20)],
            nullable=False,
            unique=True,
        ),
        "gender": Column(
            pa.String,
            checks=Check.isin(["Male", "Female"]),
            nullable=False,
        ),
        "senior_citizen": Column(
            pa.Int,
            checks=Check.isin([0, 1]),
            nullable=False,
        ),
        "partner": Column(
            pa.String,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "dependents": Column(
            pa.String,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "tenure": Column(
            pa.Int,
            checks=Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "phone_service": Column(
            pa.String,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "multiple_lines": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No phone service"]),
            nullable=False,
        ),
        "internet_service": Column(
            pa.String,
            checks=Check.isin(["DSL", "Fiber optic", "No"]),
            nullable=False,
        ),
        "online_security": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "online_backup": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "device_protection": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "tech_support": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "streaming_tv": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "streaming_movies": Column(
            pa.String,
            checks=Check.isin(["No", "Yes", "No internet service"]),
            nullable=False,
        ),
        "contract": Column(
            pa.String,
            checks=Check.isin(["Month-to-month", "One year", "Two year"]),
            nullable=False,
        ),
        "paperless_billing": Column(
            pa.String,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "payment_method": Column(
            pa.String,
            checks=Check.isin([
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]),
            nullable=False,
        ),
        "monthly_charges": Column(
            pa.Float,
            checks=Check.greater_than_or_equal_to(0.0),
            nullable=False,
        ),
        "total_charges": Column(
            pa.Float,
            checks=Check.greater_than_or_equal_to(0.0),
            nullable=False,
        ),
        "churn": Column(
            pa.String,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "churn_label": Column(
            pa.Int,
            checks=Check.isin([0, 1]),
            nullable=False,
        ),
    },
    coerce=False,
    strict=False,
)
