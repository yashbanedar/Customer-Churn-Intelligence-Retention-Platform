"""
Unit tests for data quality reporting module.
"""

import json
from pathlib import Path
import pytest

from src.data.cleaner import clean_dataset
from src.data.ingest import load_raw_data
from src.data.quality_report import generate_data_quality_report


def test_generate_data_quality_report_artifacts(tmp_path):
    """
    Verifies that generate_data_quality_report outputs complete JSON and Markdown reports.
    """
    raw_df = load_raw_data()
    clean_df = clean_dataset(raw_df)

    report = generate_data_quality_report(
        raw_df=raw_df,
        clean_df=clean_df,
        output_dir=tmp_path,
    )

    assert isinstance(report, dict)
    assert report["validation_status"] == "PASSED"
    assert "dataset_overview" in report
    assert "target_distribution" in report
    assert "missing_and_whitespace_audit" in report
    assert "numeric_summary_statistics" in report
    assert "segmented_churn_rates" in report

    # Verify JSON artifact
    json_file = tmp_path / "data_quality_report.json"
    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["dataset_overview"]["total_rows"] == 7043
    assert loaded_json["target_distribution"]["churn_rate_pct"] == 26.54

    # Verify Markdown artifact
    md_file = tmp_path / "data_quality_report.md"
    assert md_file.exists()
    md_text = md_file.read_text(encoding="utf-8")
    assert "# Data Quality & Integrity Report" in md_text
    assert "7,043" in md_text
    assert "26.54%" in md_text
