"""
Application Configuration Module
Provides type-safe, environment-aware configuration using Pydantic Settings.
Loads values from .env with fallback to sensible defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal
import os

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field
    _HAS_PYDANTIC_SETTINGS = True
except ImportError:
    _HAS_PYDANTIC_SETTINGS = False
    from pydantic import BaseModel, Field


# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent


if _HAS_PYDANTIC_SETTINGS:
    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=str(BASE_DIR / ".env"),
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False,
        )

        # Environment & Logging
        app_env: Literal["development", "staging", "production", "test"] = "development"
        app_name: str = "Telco Churn Intelligence Platform"
        log_level: str = "INFO"

        # API Configuration
        api_host: str = "127.0.0.1"
        api_port: int = 8000
        api_reload: bool = True
        api_secret_key: str = "dev-secret-key-telco-churn-retention-opt-platform"

        # Streamlit Dashboard Configuration
        dashboard_port: int = 8501
        dashboard_host: str = "127.0.0.1"

        # PostgreSQL Configuration
        postgres_user: str = "postgres"
        postgres_password: str = "postgres"
        postgres_host: str = "localhost"
        postgres_port: int = 5432
        postgres_db: str = "telco_churn_db"

        # SQLite Fallback for Local / Isolated Tests
        sqlite_db_path: str = "data/telco_churn.db"
        use_sqlite_fallback: bool = True

        # Data Source Locations
        data_raw_url: str = (
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        )
        data_raw_path: str = "data/raw/Telco-Customer-Churn.csv"
        data_processed_path: str = "data/processed/telco_churn_clean.parquet"

        # Model & Artifact Locations
        model_artifacts_dir: str = "artifacts/models"
        metrics_artifacts_dir: str = "artifacts/metrics"
        explainers_artifacts_dir: str = "artifacts/explainers"
        active_model_path: str = "artifacts/models/calibrated_xgb_pipeline.joblib"

        # Business Retention Simulator Defaults
        default_campaign_budget: float = 50000.0
        default_offer_cost: float = 35.0
        default_save_rate: float = 0.22
        default_horizon_months: int = 12

        @property
        def database_url(self) -> str:
            """
            Returns the active SQLAlchemy database URL based on environment and fallback flags.
            """
            if self.use_sqlite_fallback or self.app_env == "test":
                db_file = BASE_DIR / self.sqlite_db_path
                db_file.parent.mkdir(parents=True, exist_ok=True)
                return f"sqlite:///{db_file.as_posix()}"
            return (
                f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )

        @property
        def raw_data_file(self) -> Path:
            return BASE_DIR / self.data_raw_path

        @property
        def processed_data_file(self) -> Path:
            return BASE_DIR / self.data_processed_path

else:
    # Lightweight fallback if pydantic-settings is not yet installed in active python
    from pydantic import BaseModel

    class Settings(BaseModel):
        app_env: str = os.getenv("APP_ENV", "development")
        app_name: str = os.getenv("APP_NAME", "Telco Churn Intelligence Platform")
        log_level: str = os.getenv("LOG_LEVEL", "INFO")
        api_host: str = os.getenv("API_HOST", "127.0.0.1")
        api_port: int = int(os.getenv("API_PORT", "8000"))
        api_reload: bool = os.getenv("API_RELOAD", "True").lower() in ("true", "1")
        api_secret_key: str = os.getenv("API_SECRET_KEY", "dev-secret-key-telco-churn-retention-opt-platform")
        dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8501"))
        dashboard_host: str = os.getenv("DASHBOARD_HOST", "127.0.0.1")
        postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
        postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")
        postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
        postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
        postgres_db: str = os.getenv("POSTGRES_DB", "telco_churn_db")
        sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", "data/telco_churn.db")
        use_sqlite_fallback: bool = os.getenv("USE_SQLITE_FALLBACK", "True").lower() in ("true", "1")
        data_raw_url: str = os.getenv(
            "DATA_RAW_URL",
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        )
        data_raw_path: str = os.getenv("DATA_RAW_PATH", "data/raw/Telco-Customer-Churn.csv")
        data_processed_path: str = os.getenv("DATA_PROCESSED_PATH", "data/processed/telco_churn_clean.parquet")
        model_artifacts_dir: str = os.getenv("MODEL_ARTIFACTS_DIR", "artifacts/models")
        metrics_artifacts_dir: str = os.getenv("METRICS_ARTIFACTS_DIR", "artifacts/metrics")
        explainers_artifacts_dir: str = os.getenv("EXPLAINERS_ARTIFACTS_DIR", "artifacts/explainers")
        active_model_path: str = os.getenv("ACTIVE_MODEL_PATH", "artifacts/models/calibrated_xgb_pipeline.joblib")
        default_campaign_budget: float = float(os.getenv("DEFAULT_CAMPAIGN_BUDGET", "50000.0"))
        default_offer_cost: float = float(os.getenv("DEFAULT_OFFER_COST", "35.0"))
        default_save_rate: float = float(os.getenv("DEFAULT_SAVE_RATE", "0.22"))
        default_horizon_months: int = int(os.getenv("DEFAULT_HORIZON_MONTHS", "12"))

        @property
        def database_url(self) -> str:
            if self.use_sqlite_fallback or self.app_env == "test":
                db_file = BASE_DIR / self.sqlite_db_path
                db_file.parent.mkdir(parents=True, exist_ok=True)
                return f"sqlite:///{db_file.as_posix()}"
            return (
                f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )

        @property
        def raw_data_file(self) -> Path:
            return BASE_DIR / self.data_raw_path

        @property
        def processed_data_file(self) -> Path:
            return BASE_DIR / self.data_processed_path


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of application settings.
    """
    return Settings()
