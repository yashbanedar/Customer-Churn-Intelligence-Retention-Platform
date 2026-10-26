"""
FastAPI Microservice Integration and Input Validation Tests (Phase 7).
Tests health endpoint, single prediction, batch prediction, Pydantic validation errors,
and database analytical summary queries using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient triggering app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer_payload():
    """Deterministic valid customer payload for testing."""
    return {
        "customer_id": "TEST-CUST-001",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 1,
        "phone_service": "No",
        "multiple_lines": "No phone service",
        "internet_service": "DSL",
        "online_security": "No",
        "online_backup": "Yes",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 29.85,
        "total_charges": 29.85,
    }


class TestAPIBasicsAndHealth:
    """Tests general endpoints and health status check."""

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "documentation" in data
        assert data["version"] == "1.0.0"

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert data["model_loaded"] is True
        assert data["active_threshold"] == 0.540
        assert "database_connected" in data
        assert isinstance(data["database_tables"], list)


class TestSinglePredictionValidation:
    """Tests /predict input validation constraints and inference outputs."""

    def test_predict_single_success(self, client, valid_customer_payload):
        response = client.post("/predict", json=valid_customer_payload)
        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == "TEST-CUST-001"
        assert 0.0 <= data["calibrated_churn_probability"] <= 1.0
        assert isinstance(data["predicted_churn"], bool)
        assert data["decision_threshold"] == 0.540
        assert data["risk_tier"] in ("High Risk", "Medium Risk", "Low Risk")
        assert "expected_net_value" in data
        assert data["recommended_action"] in (
            "Proactive Retention Outreach",
            "Standard Engagement",
            "Loyal / Monitor",
        )
        assert len(data["key_drivers"]) >= 1

    def test_predict_validation_negative_tenure(self, client, valid_customer_payload):
        payload = valid_customer_payload.copy()
        payload["tenure"] = -5
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert data["error"] == "Validation Error"
        assert any("tenure" in d["field"] for d in data["details"])

    def test_predict_validation_invalid_categorical(self, client, valid_customer_payload):
        payload = valid_customer_payload.copy()
        payload["contract"] = "Unlimited Century"  # Invalid contract
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert any("contract" in d["field"] for d in data["details"])

    def test_predict_validation_negative_charges(self, client, valid_customer_payload):
        payload = valid_customer_payload.copy()
        payload["monthly_charges"] = -50.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_validation_missing_field(self, client, valid_customer_payload):
        payload = valid_customer_payload.copy()
        del payload["payment_method"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Tests /predict/batch endpoint with multiple customer records."""

    def test_predict_batch_success(self, client, valid_customer_payload):
        p1 = valid_customer_payload.copy()
        p1["customer_id"] = "CUST-001"
        p2 = valid_customer_payload.copy()
        p2["customer_id"] = "CUST-002"
        p2["contract"] = "Two year"
        p2["tenure"] = 48

        batch_payload = {
            "customers": [p1, p2],
            "custom_threshold": 0.540,
            "offer_cost": 35.0,
            "assumed_save_rate": 0.22,
            "customer_clv": 450.0,
        }
        response = client.post("/predict/batch", json=batch_payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2
        assert data["summary"]["total_customers"] == 2
        assert 0.0 <= data["summary"]["mean_calibrated_probability"] <= 1.0
        assert "causal_disclaimer" in data

    def test_predict_batch_empty_list_rejected(self, client):
        batch_payload = {"customers": []}
        response = client.post("/predict/batch", json=batch_payload)
        assert response.status_code == 422

    def test_predict_batch_invalid_threshold_rejected(self, client, valid_customer_payload):
        batch_payload = {
            "customers": [valid_customer_payload],
            "custom_threshold": 1.5,  # Must be <= 0.95
        }
        response = client.post("/predict/batch", json=batch_payload)
        assert response.status_code == 422


class TestAnalyticsSummaryEndpoint:
    """Tests /analytics/summary database integration."""

    def test_analytics_summary_success(self, client):
        response = client.get("/analytics/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_customers"] > 0
        assert 0.0 <= data["churn_rate_pct"] <= 100.0
        assert data["total_mrr"] > 0
        assert len(data["contracts_breakdown"]) >= 1
        assert len(data["tenure_cohorts"]) >= 1
