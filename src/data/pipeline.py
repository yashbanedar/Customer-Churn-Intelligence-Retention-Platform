"""
End-to-End Data Pipeline Orchestrator.
Coordinates dataset ingestion, schema validation, cleaning, quality reporting,
and relational database population.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Union

from src.config import get_settings
from src.data.cleaner import clean_dataset, save_processed_data
from src.data.db_loader import load_dataset_to_sql
from src.data.ingest import fetch_dataset, load_raw_data
from src.data.quality_report import generate_data_quality_report
from src.data.validator import DataValidator


logger = logging.getLogger("src.data.pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_pipeline(
    download: bool = True,
    force_download: bool = False,
    validate_only: bool = False,
    report_only: bool = False,
    load_db: bool = True,
    raw_path: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Executes the comprehensive Phase 2 Data Pipeline:
    1. Ingestion: Download / load raw dataset.
    2. Validation: Validate raw schema, types, duplicates, whitespace, domains, target labels.
    3. Cleaning: Normalize columns, cast types, impute TotalCharges spaces.
    4. Validation: Verify cleaned dataset against Pandera Clean Schema.
    5. Quality Report: Generate structured JSON and Markdown audit reports.
    6. Database Load: Populate normalized PostgreSQL / SQLite tables.

    Returns:
        True if pipeline succeeded, False otherwise.
    """
    settings = get_settings()
    logger.info("=" * 70)
    logger.info("STARTING IBM TELCO CUSTOMER CHURN DATA INGESTION & VALIDATION PIPELINE")
    logger.info("=" * 70)

    # Step 1: Ingestion
    if download:
        raw_file = fetch_dataset(force_download=force_download, save_path=raw_path)
    else:
        raw_file = Path(raw_path) if raw_path else settings.raw_data_file

    if not raw_file.exists():
        logger.error(f"Raw data file not found at: {raw_file}. Enable --download to fetch.")
        return False

    raw_df = load_raw_data(raw_file)

    # Step 2: Raw Data Validation
    logger.info("--> STEP 2: Running Rigorous Raw Data Validation...")
    validator = DataValidator()
    raw_val = validator.validate_raw(raw_df)

    if not raw_val.is_valid:
        logger.error("Raw data validation failed! Errors detected:")
        for err in raw_val.errors:
            logger.error(f"  ❌ {err}")
        if not report_only:
            return False
    else:
        logger.info("✅ Raw data validation passed with 0 fatal errors.")

    if validate_only:
        logger.info("Validation complete (--validate-only flag provided). Exiting successfully.")
        return True

    # Step 3: Cleaning & Transformation
    logger.info("--> STEP 3: Transforming and Cleaning Dataset...")
    clean_df = clean_dataset(raw_df, impute_zero_tenure=True)

    # Step 4: Clean Data Validation
    logger.info("--> STEP 4: Validating Cleaned Dataset Schema...")
    clean_val = validator.validate_clean(clean_df)
    if not clean_val.is_valid:
        logger.error("Cleaned data validation failed!")
        for err in clean_val.errors:
            logger.error(f"  ❌ {err}")
        return False
    logger.info("✅ Cleaned dataset verified against Pandera production contract.")

    # Save processed artifacts
    save_processed_data(clean_df)

    # Step 5: Data Quality Report Generation
    logger.info("--> STEP 5: Generating Comprehensive Data Quality Report...")
    report = generate_data_quality_report(raw_df=raw_df, clean_df=clean_df)
    logger.info(
        f"✅ Data quality report generated. Churn Rate: {report['target_distribution']['churn_rate_pct']}%. Status: {report['validation_status']}."
    )

    if report_only:
        return True

    # Step 6: Load to Relational Database
    if load_db:
        logger.info("--> STEP 6: Loading Normalized Relational Tables into Database...")
        db_counts = load_dataset_to_sql(clean_df, recreate_tables=True)
        logger.info(f"✅ Database loaded successfully. Record counts: {db_counts}")

    logger.info("=" * 70)
    logger.info("PHASE 2 DATA INGESTION & VALIDATION PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description="IBM Telco Churn Data Ingestion & Validation Pipeline.")
    parser.add_argument("--download", action="store_true", default=True, help="Download dataset if missing.")
    parser.add_argument("--force-download", action="store_true", help="Force redownload even if file exists.")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation checks on raw data.")
    parser.add_argument("--report-only", action="store_true", help="Run ingestion, validation, and report only.")
    parser.add_argument("--no-db", action="store_true", help="Skip populating relational database tables.")
    parser.add_argument("--raw-path", type=str, help="Custom path to raw CSV file.")

    args = parser.parse_args()

    success = run_pipeline(
        download=args.download,
        force_download=args.force_download,
        validate_only=args.validate_only,
        report_only=args.report_only,
        load_db=not args.no_db,
        raw_path=args.raw_path,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
