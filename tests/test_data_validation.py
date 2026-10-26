"""
Unit tests for data validation module:
Tests column presence, duplicate detection, whitespace handling, target labels,
categorical domains, numeric bounds, and Pandera contracts.
"""

import pandas as pd
import pytest

from src.data.ingest import load_raw_data
from src.data.validator import DataValidator


@pytest.fixture
def base_sample_df() -> pd.DataFrame:
    """Returns a valid 5-row raw DataFrame slice."""
    df = load_raw_data()
    return df.head(5).copy()


def test_validator_passes_on_valid_raw_data(base_sample_df):
    """
    Valid slice of raw data passes validation checks.
    """
    validator = DataValidator()
    result = validator.validate_raw(base_sample_df)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.checks_passed > 0


def test_validator_detects_duplicate_customer_ids(base_sample_df):
    """
    Duplicate customerID records must be flagged as fatal validation errors.
    """
    bad_df = base_sample_df.copy()
    # Duplicate first row's customerID on second row
    bad_df.iloc[1, bad_df.columns.get_loc("customerID")] = bad_df.iloc[0]["customerID"]

    validator = DataValidator()
    result = validator.validate_raw(bad_df)
    assert result.is_valid is False
    assert any("duplicate customerID" in err for err in result.errors)
    assert result.metrics["duplicate_customer_ids"] == 1


def test_validator_detects_missing_required_column(base_sample_df):
    """
    Missing required column (e.g. Churn) must fail validation immediately.
    """
    bad_df = base_sample_df.drop(columns=["Churn"])
    validator = DataValidator()
    result = validator.validate_raw(bad_df)
    assert result.is_valid is False
    assert any("Missing required columns" in err for err in result.errors)


def test_validator_detects_invalid_churn_target_label(base_sample_df):
    """
    Target labels other than 'Yes' or 'No' must trigger validation errors.
    """
    bad_df = base_sample_df.copy()
    bad_df.iloc[0, bad_df.columns.get_loc("Churn")] = "Maybe"

    validator = DataValidator()
    result = validator.validate_raw(bad_df)
    assert result.is_valid is False
    assert any("Invalid target labels in 'Churn'" in err for err in result.errors)


def test_validator_detects_invalid_categorical_domain(base_sample_df):
    """
    Invalid values in categorical fields (e.g. InternetService = 'Satellite_5G') must fail validation.
    """
    bad_df = base_sample_df.copy()
    bad_df.iloc[0, bad_df.columns.get_loc("InternetService")] = "Satellite_5G"

    validator = DataValidator()
    result = validator.validate_raw(bad_df)
    assert result.is_valid is False
    assert any("Categorical domain violations" in err for err in result.errors)


def test_validator_detects_negative_numeric_values(base_sample_df):
    """
    Negative values in tenure or MonthlyCharges must trigger validation errors.
    """
    bad_df = base_sample_df.copy()
    bad_df.iloc[0, bad_df.columns.get_loc("tenure")] = -5

    validator = DataValidator()
    result = validator.validate_raw(bad_df)
    assert result.is_valid is False
    assert any("negative values in 'tenure'" in err for err in result.errors)


def test_validator_detects_invalid_senior_citizen_values(base_sample_df):
    """
    SeniorCitizen values outside {0, 1} must trigger validation error.
    """
    bad_df = base_sample_df.copy()
    bad_df.iloc[0, bad_df.columns.get_loc("SeniorCitizen")] = 99

    validator = DataValidator()
    result = validator.validate_raw(bad_df)
    assert result.is_valid is False
    assert any("SeniorCitizen must be in {0, 1}" in err for err in result.errors)


def test_validator_flags_unexplained_whitespace_in_total_charges(base_sample_df):
    """
    Whitespace in TotalCharges where tenure > 0 must trigger an error.
    """
    bad_df = base_sample_df.copy()
    bad_df.iloc[0, bad_df.columns.get_loc("TotalCharges")] = "   "
    bad_df.iloc[0, bad_df.columns.get_loc("tenure")] = 12  # Tenure is 12, not 0!

    validator = DataValidator()
    result = validator.validate_raw(bad_df)
    assert result.is_valid is False
    assert any("whitespace in 'TotalCharges'" in err for err in result.errors)
