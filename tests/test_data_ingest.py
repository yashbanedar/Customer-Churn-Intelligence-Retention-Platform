"""
Unit tests for data ingestion module.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data.ingest import fetch_dataset, load_raw_data
from src.data.schema import RAW_COLUMNS


def test_load_raw_data_structure():
    """
    Verifies that load_raw_data returns a valid DataFrame matching all 21 raw IBM Telco columns.
    """
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7043
    assert len(df.columns) == 21
    for col in RAW_COLUMNS:
        assert col in df.columns, f"Expected raw column '{col}' not found."


def test_fetch_dataset_idempotent(tmp_path):
    """
    Verifies that fetch_dataset does not re-download if file already exists with content.
    """
    dummy_file = tmp_path / "dummy_raw.csv"
    dummy_file.write_text("customerID,Churn\n1234-ABCDE,No\n", encoding="utf-8")

    # Should return existing file without making network request
    result_path = fetch_dataset(save_path=dummy_file, force_download=False)
    assert result_path == dummy_file
    assert dummy_file.read_text(encoding="utf-8").startswith("customerID,Churn")
