"""
Global Pytest Fixtures and Configurations
"""

import pytest
import os
from src.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clean_settings_cache():
    """
    Clears the lru_cache of get_settings between tests to avoid cross-test contamination.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_settings() -> Settings:
    """
    Provides isolated test settings configured for test environment.
    """
    return Settings(
        app_env="test",
        use_sqlite_fallback=True,
        sqlite_db_path="data/test_telco_churn.db",
        log_level="DEBUG",
    )


@pytest.fixture
def sample_customer_payload() -> dict:
    """
    Returns a standard test customer record matching the IBM Telco schema.
    """
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
        "Churn": "No",
    }
