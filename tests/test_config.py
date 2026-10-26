"""
Unit Tests for Configuration Management and Environment Handling
"""

from pathlib import Path
from src.config import Settings, get_settings, BASE_DIR


def test_settings_initialization_defaults():
    """
    Test that Settings loads valid default values without error.
    """
    settings = get_settings()
    assert settings.app_name == "Telco Churn Intelligence Platform"
    assert settings.api_port == 8000
    assert settings.dashboard_port == 8501
    assert settings.default_campaign_budget > 0
    assert 0.0 < settings.default_save_rate < 1.0


def test_database_url_resolution():
    """
    Test that database_url returns a valid connection URI.
    """
    settings = get_settings()
    db_url = settings.database_url
    assert isinstance(db_url, str)
    # If fallback is active or test env, it should point to sqlite
    if settings.use_sqlite_fallback or settings.app_env == "test":
        assert db_url.startswith("sqlite:///")
    else:
        assert db_url.startswith("postgresql+psycopg2://")


def test_directory_paths_resolve_to_project_root():
    """
    Test that raw_data_file and processed_data_file paths are anchored to project BASE_DIR.
    """
    settings = get_settings()
    raw_path = settings.raw_data_file
    processed_path = settings.processed_data_file

    assert isinstance(raw_path, Path)
    assert isinstance(processed_path, Path)
    assert raw_path.name == "Telco-Customer-Churn.csv"
    assert processed_path.name == "telco_churn_clean.parquet"
    assert str(BASE_DIR) in str(raw_path)


def test_test_settings_fixture(test_settings):
    """
    Test that the test_settings fixture correctly isolates test environment.
    """
    assert test_settings.app_env == "test"
    assert test_settings.use_sqlite_fallback is True
    assert "sqlite:///" in test_settings.database_url
