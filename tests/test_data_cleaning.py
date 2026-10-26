"""
Unit tests for data cleaning and transformation module.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import clean_dataset, save_processed_data
from src.data.ingest import load_raw_data


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Provides a sample slice including zero tenure records."""
    df = load_raw_data()
    # Ensure we include a zero-tenure record
    zero_tenure = df[df["tenure"] == 0].head(2)
    regular = df[df["tenure"] > 0].head(3)
    return pd.concat([zero_tenure, regular], ignore_index=True)


def test_clean_dataset_renames_to_snake_case(sample_raw_df):
    """
    Verifies that columns are transformed to standardized snake_case.
    """
    clean_df = clean_dataset(sample_raw_df)
    expected_cols = [
        "customer_id",
        "gender",
        "senior_citizen",
        "partner",
        "dependents",
        "tenure",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "contract",
        "paperless_billing",
        "payment_method",
        "monthly_charges",
        "total_charges",
        "churn",
        "churn_label",
    ]
    for c in expected_cols:
        assert c in clean_df.columns, f"Expected column '{c}' not found in clean dataframe."


def test_clean_dataset_imputes_zero_tenure_total_charges(sample_raw_df):
    """
    Verifies that whitespace TotalCharges at tenure=0 is imputed with 0.0 float.
    """
    clean_df = clean_dataset(sample_raw_df, impute_zero_tenure=True)
    zero_tenure_rows = clean_df[clean_df["tenure"] == 0]
    assert len(zero_tenure_rows) > 0
    assert (zero_tenure_rows["total_charges"] == 0.0).all()
    assert np.issubdtype(clean_df["total_charges"].dtype, np.floating)
    assert clean_df["total_charges"].isna().sum() == 0


def test_clean_dataset_creates_binary_churn_label(sample_raw_df):
    """
    Verifies that churn_label matches 'Yes' -> 1 and 'No' -> 0.
    """
    clean_df = clean_dataset(sample_raw_df)
    assert "churn_label" in clean_df.columns
    assert set(clean_df["churn_label"].unique()).issubset({0, 1})

    for _, row in clean_df.iterrows():
        if row["churn"] == "Yes":
            assert row["churn_label"] == 1
        elif row["churn"] == "No":
            assert row["churn_label"] == 0


def test_save_processed_data_creates_parquet_and_csv(sample_raw_df, tmp_path):
    """
    Verifies that save_processed_data creates valid Parquet and CSV files on disk.
    """
    clean_df = clean_dataset(sample_raw_df)
    parquet_dest = tmp_path / "test_clean.parquet"
    csv_dest = tmp_path / "test_clean.csv"

    out_file = save_processed_data(clean_df, parquet_path=parquet_dest, csv_path=csv_dest)
    assert out_file == parquet_dest
    assert parquet_dest.exists() and parquet_dest.stat().st_size > 0
    assert csv_dest.exists() and csv_dest.stat().st_size > 0

    # Verify reload from parquet
    reloaded_df = pd.read_parquet(parquet_dest)
    assert len(reloaded_df) == len(clean_df)
    assert "churn_label" in reloaded_df.columns
