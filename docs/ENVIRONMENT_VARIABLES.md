# Environment Variables Reference

This document provides a complete specification of all configuration settings and environment variables used across the **Customer Churn Intelligence & Retention Optimization Platform**.

All settings are managed via **Pydantic Settings** in [`src/config.py`](file:///c:/Users/prati/.gemini/antigravity/scratch/Customer%20Churn%20Intelligence%20&%20Retention%20Optimization%20Platform/src/config.py), enabling automatic type coercion, default fallback, and validation.

---

## 📋 Configuration Matrix

| Variable Name | Type | Default Value | Production Required? | Description & Security Guidance |
| :--- | :---: | :---: | :---: | :--- |
| `APP_ENV` | `str` | `"development"` | No | Application environment: `development`, `staging`, `production`, `test`. |
| `APP_NAME` | `str` | `"Telco Churn Intelligence Platform"` | No | Display name used in OpenAPI docs and headers. |
| `LOG_LEVEL` | `str` | `"INFO"` | No | Python standard logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `API_HOST` | `str` | `"127.0.0.1"` | No | Host interface for FastAPI ASGI server (`0.0.0.0` in Docker). |
| `API_PORT` | `int` | `8000` | No | Listening port for FastAPI microservice. |
| `API_RELOAD` | `bool` | `True` | No | Enables hot-reloading in development. Set to `False` in production! |
| `API_SECRET_KEY` | `str` | `"dev-secret-key-..."` | **YES** | Cryptographic key for API signing. **Must be rotated in production.** |
| `DASHBOARD_HOST` | `str` | `"127.0.0.1"` | No | Host interface for Streamlit dashboard. |
| `DASHBOARD_PORT` | `int` | `8501` | No | Listening port for Streamlit executive UI. |
| `POSTGRES_USER` | `str` | `"postgres"` | **YES** | Database username for PostgreSQL relational store. |
| `POSTGRES_PASSWORD` | `str` | `"postgres"` | **YES** | Database password. **Never commit real credentials to Git!** |
| `POSTGRES_HOST` | `str` | `"localhost"` | **YES** | PostgreSQL hostname or container service name (`postgres` in Docker). |
| `POSTGRES_PORT` | `int` | `5432` | No | PostgreSQL standard listening port. |
| `POSTGRES_DB` | `str` | `"telco_churn_db"` | No | Target relational database name. |
| `SQLITE_DB_PATH` | `str` | `"data/telco_churn.db"` | No | Relative path to local SQLite database file for offline testing. |
| `USE_SQLITE_FALLBACK` | `bool` | `True` | No | When `True`, routes queries to local SQLite if PostgreSQL is absent. |
| `DATA_RAW_URL` | `str` | Canonical IBM GitHub URL | No | Remote URL to fetch raw IBM Telco Customer Churn CSV. |
| `DATA_RAW_PATH` | `str` | `"data/raw/Telco-Customer-Churn.csv"` | No | Local path to cache the downloaded raw dataset. |
| `DATA_PROCESSED_PATH`| `str` | `"data/processed/telco_churn_clean.parquet"` | No | Path to validated and cleaned Parquet dataset. |
| `MODEL_ARTIFACTS_DIR`| `str` | `"artifacts/models"` | No | Directory where fitted scikit-learn pipelines are persisted. |
| `METRICS_ARTIFACTS_DIR`| `str` | `"artifacts/metrics"` | No | Directory for benchmark JSON, markdown reports, and quality audits. |
| `EXPLAINERS_ARTIFACTS_DIR`| `str` | `"artifacts/explainers"` | No | Directory where serialized SHAP explainers are stored. |
| `ACTIVE_MODEL_PATH` | `str` | `"artifacts/models/calibrated_xgb_pipeline.joblib"` | No | Path to active model pipeline loaded by FastAPI and Streamlit. |
| `DEFAULT_CAMPAIGN_BUDGET`| `float` | `50000.0` | No | Default budget allocation for retention campaigns ($). |
| `DEFAULT_OFFER_COST` | `float` | `35.0` | No | Default cost per proactive customer intervention ($). |
| `DEFAULT_SAVE_RATE` | `float` | `0.22` | No | Default assumed counterfactual offer save rate (22%). |
| `DEFAULT_HORIZON_MONTHS`| `int` | `12` | No | Default planning horizon for customer lifetime value (months). |

---

## 🔒 Security Best Practices

1. **Local Development**:
   Copy `.env.example` to `.env`. Leave `USE_SQLITE_FALLBACK=True` to run completely offline without needing a running PostgreSQL instance.
2. **Docker Compose**:
   Set `USE_SQLITE_FALLBACK=False` so containers automatically connect to the `postgres` service on the internal `telco_network` bridge.
3. **Cloud Production**:
   Do NOT commit `.env` to source control. Inject environment variables via your cloud orchestrator's secret management system (e.g. AWS Secrets Manager, GitHub Actions Secrets, or Kubernetes Secrets).
