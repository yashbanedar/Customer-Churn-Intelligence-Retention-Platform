"""
Data transformation and cleaning module for IBM Telco Customer Churn dataset.
Handles column renaming, type casting, missing value imputation (zero tenure TotalCharges),
and exporting to Parquet and CSV.
"""

import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from src.config import get_settings
from src.data.schema import RAW_TO_SNAKE_CASE
from src.data.validator import DataValidator


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def clean_dataset(
    df: pd.DataFrame,
    impute_zero_tenure: bool = True,
) -> pd.DataFrame:
    """
    Cleans raw IBM Telco DataFrame into production-ready analytical format:
    1. Standardizes columns to snake_case.
    2. Converts total_charges to float, imputing zero-tenure whitespace records with 0.0.
    3. Synthesizes binary target churn_label (1 for Yes, 0 for No).
    4. Enforces strict numeric and categorical dtypes.

    Args:
        df: Raw DataFrame matching IBM Telco schema.
        impute_zero_tenure: If True, sets total_charges=0.0 when tenure=0.

    Returns:
        Cleaned, type-safe pandas DataFrame.
    """
    logger.info("Initiating dataset cleaning and schema normalization...")
    clean_df = df.copy()

    # 1. Rename columns to snake_case
    clean_df = clean_df.rename(columns=RAW_TO_SNAKE_CASE)

    # 2. Handle TotalCharges numeric coercion
    if "total_charges" in clean_df.columns:
        # Convert empty strings / spaces to NaN, then float
        clean_df["total_charges"] = pd.to_numeric(
            clean_df["total_charges"].astype(str).str.strip(), errors="coerce"
        )

        if impute_zero_tenure:
            zero_tenure_mask = (clean_df["tenure"] == 0) & (clean_df["total_charges"].isna())
            imputed_count = int(zero_tenure_mask.sum())
            if imputed_count > 0:
                logger.info(f"Imputing total_charges = 0.0 for {imputed_count} accounts with tenure = 0.")
                clean_df.loc[zero_tenure_mask, "total_charges"] = 0.0

    # 3. Create binary churn_label
    if "churn" in clean_df.columns:
        clean_df["churn_label"] = (clean_df["churn"] == "Yes").astype(int)

    # 4. Cast numeric types
    clean_df["tenure"] = clean_df["tenure"].astype(int)
    clean_df["senior_citizen"] = clean_df["senior_citizen"].astype(int)
    clean_df["monthly_charges"] = clean_df["monthly_charges"].astype(float)
    clean_df["total_charges"] = clean_df["total_charges"].astype(float)

    # 5. Trim string columns
    str_cols = clean_df.select_dtypes(include=["object", "string", "str"]).columns
    for c in str_cols:
        clean_df[c] = clean_df[c].astype(str).str.strip()

    logger.info(f"Cleaning complete. Output shape: {clean_df.shape}")
    return clean_df


def save_processed_data(
    df: pd.DataFrame,
    parquet_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Persists cleaned DataFrame to disk in Parquet and CSV formats.

    Args:
        df: Cleaned DataFrame.
        parquet_path: Destination parquet file path. Defaults to settings.data_processed_path.
        csv_path: Destination CSV file path. Defaults to same directory as parquet.

    Returns:
        Path to the primary saved Parquet file.
    """
    settings = get_settings()
    p_path = Path(parquet_path) if parquet_path else settings.processed_data_file
    p_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save Parquet
    df.to_parquet(p_path, engine="pyarrow", index=False)
    logger.info(f"Saved processed Parquet to: {p_path} ({p_path.stat().st_size:,} bytes)")

    # 2. Save CSV alongside for downstream interoperability
    c_path = Path(csv_path) if csv_path else p_path.with_suffix(".csv")
    df.to_csv(c_path, index=False)
    logger.info(f"Saved processed CSV to: {c_path} ({c_path.stat().st_size:,} bytes)")

    return p_path
