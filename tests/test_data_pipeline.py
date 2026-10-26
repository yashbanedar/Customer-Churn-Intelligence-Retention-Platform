"""
Unit tests for data pipeline orchestrator.
"""

from pathlib import Path
import pytest

from src.data.pipeline import run_pipeline


def test_run_pipeline_validate_only():
    """
    Verifies that running pipeline with validate_only=True executes and returns True.
    """
    success = run_pipeline(download=False, validate_only=True, load_db=False)
    assert success is True


def test_run_pipeline_report_only(tmp_path):
    """
    Verifies that running pipeline with report_only=True runs ingestion, validation, cleaning, and reporting.
    """
    success = run_pipeline(download=False, report_only=True, load_db=False)
    assert success is True


def test_run_pipeline_end_to_end():
    """
    Verifies full pipeline execution with DB loading on existing raw data.
    """
    success = run_pipeline(download=False, validate_only=False, load_db=True)
    assert success is True
