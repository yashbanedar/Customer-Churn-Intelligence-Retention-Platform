"""
Data ingestion, validation, cleaning, quality reporting, and database loading package.
"""

from src.data.cleaner import clean_dataset, save_processed_data
from src.data.db_loader import execute_schema_ddl, get_db_engine, load_dataset_to_sql
from src.data.ingest import fetch_dataset, load_raw_data
from src.data.quality_report import generate_data_quality_report
from src.data.schema import (
    CATEGORICAL_DOMAINS,
    CLEAN_DATA_SCHEMA,
    RAW_COLUMNS,
    RAW_DATA_SCHEMA,
    RAW_TO_SNAKE_CASE,
)
from src.data.validator import DataValidator, ValidationResult

__all__ = [
    "fetch_dataset",
    "load_raw_data",
    "clean_dataset",
    "save_processed_data",
    "DataValidator",
    "ValidationResult",
    "generate_data_quality_report",
    "load_dataset_to_sql",
    "get_db_engine",
    "execute_schema_ddl",
    "RAW_COLUMNS",
    "RAW_TO_SNAKE_CASE",
    "CATEGORICAL_DOMAINS",
    "RAW_DATA_SCHEMA",
    "CLEAN_DATA_SCHEMA",
]
