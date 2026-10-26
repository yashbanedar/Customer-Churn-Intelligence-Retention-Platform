"""
Data validation module for IBM Telco Customer Churn dataset.
Performs rigorous structural, type, duplicate, domain, target label, and relational checks.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pandera.errors

from src.data.schema import (
    CATEGORICAL_DOMAINS,
    CLEAN_CATEGORICAL_DOMAINS,
    CLEAN_DATA_SCHEMA,
    RAW_COLUMNS,
    RAW_DATA_SCHEMA,
)


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class ValidationResult:
    """
    Structured outcome of a data validation run.
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"ValidationResult({status}): {self.checks_passed} checks passed, "
            f"{self.checks_failed} checks failed. "
            f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}"
        )


class DataValidator:
    """
    Validates raw and cleaned customer churn datasets against business logic and schema rules.
    """

    def validate_raw(self, df: pd.DataFrame) -> ValidationResult:
        """
        Runs comprehensive validation on the raw IBM Telco DataFrame.

        Checks performed:
        1. Required column presence
        2. Duplicate primary keys (customerID)
        3. Missing values (null / NaN)
        4. Whitespace string detection (specifically TotalCharges)
        5. Target label validation (Churn in {'Yes', 'No'})
        6. Categorical domain compliance
        7. Numeric bounds (tenure >= 0, MonthlyCharges >= 0)
        8. Service relational consistency
        9. Pandera schema contract
        """
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {}
        passed = 0
        failed = 0

        total_rows = len(df)
        metrics["total_records"] = total_rows
        metrics["total_columns"] = len(df.columns)

        # 1. Column Presence Check
        missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns in raw dataset: {missing_cols}")
            failed += 1
        else:
            passed += 1

        # Return early if fundamental columns are missing
        if "customerID" not in df.columns or "Churn" not in df.columns:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                checks_passed=passed,
                checks_failed=failed,
                metrics=metrics,
            )

        # 2. Duplicate Primary Key Check
        duplicate_count = int(df["customerID"].duplicated().sum())
        metrics["duplicate_customer_ids"] = duplicate_count
        if duplicate_count > 0:
            errors.append(f"Found {duplicate_count} duplicate customerID records.")
            failed += 1
        else:
            passed += 1

        # 3. Missing / Null Values Check
        null_counts = df.isnull().sum().to_dict()
        columns_with_nulls = {k: v for k, v in null_counts.items() if v > 0}
        metrics["null_counts"] = columns_with_nulls
        if columns_with_nulls:
            errors.append(f"Unexpected null/NaN values found in columns: {columns_with_nulls}")
            failed += 1
        else:
            passed += 1

        # 4. Whitespace Detection (TotalCharges)
        if "TotalCharges" in df.columns:
            # Check for strings with only spaces
            whitespace_mask = df["TotalCharges"].astype(str).str.strip() == ""
            whitespace_count = int(whitespace_mask.sum())
            metrics["whitespace_total_charges_count"] = whitespace_count
            if whitespace_count > 0:
                # Check if all whitespace records have tenure == 0
                zero_tenure_whitespace = int(
                    (whitespace_mask & (pd.to_numeric(df["tenure"], errors="coerce") == 0)).sum()
                )
                if zero_tenure_whitespace == whitespace_count:
                    warnings.append(
                        f"Detected {whitespace_count} rows with whitespace in 'TotalCharges'. "
                        f"All correspond to new accounts with tenure=0 (expected IBM dataset quirk)."
                    )
                    passed += 1
                else:
                    errors.append(
                        f"Detected {whitespace_count} rows with whitespace in 'TotalCharges', "
                        f"but only {zero_tenure_whitespace} have tenure=0."
                    )
                    failed += 1
            else:
                passed += 1

        # 5. Target Label Validation (Churn)
        unique_targets = set(df["Churn"].dropna().unique())
        allowed_targets = {"Yes", "No"}
        metrics["target_classes"] = list(unique_targets)
        metrics["churn_distribution"] = df["Churn"].value_counts().to_dict()

        if not unique_targets.issubset(allowed_targets):
            invalid_targets = unique_targets - allowed_targets
            errors.append(f"Invalid target labels in 'Churn': {invalid_targets}. Allowed: {allowed_targets}")
            failed += 1
        else:
            passed += 1

        # 6. Categorical Domain Validation
        domain_failures = {}
        for col, valid_values in CATEGORICAL_DOMAINS.items():
            if col in df.columns:
                actual_values = set(df[col].dropna().unique())
                disallowed = actual_values - valid_values
                if disallowed:
                    domain_failures[col] = list(disallowed)

        if domain_failures:
            errors.append(f"Categorical domain violations detected: {domain_failures}")
            failed += 1
        else:
            passed += 1

        # 7. Numeric Bounds Validation
        if "tenure" in df.columns:
            tenure_numeric = pd.to_numeric(df["tenure"], errors="coerce")
            neg_tenure = int((tenure_numeric < 0).sum())
            if neg_tenure > 0:
                errors.append(f"Found {neg_tenure} negative values in 'tenure'.")
                failed += 1
            else:
                passed += 1

        if "MonthlyCharges" in df.columns:
            monthly_numeric = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
            neg_monthly = int((monthly_numeric < 0).sum())
            if neg_monthly > 0:
                errors.append(f"Found {neg_monthly} negative values in 'MonthlyCharges'.")
                failed += 1
            else:
                passed += 1

        if "SeniorCitizen" in df.columns:
            senior_vals = set(df["SeniorCitizen"].unique())
            if not senior_vals.issubset({0, 1}):
                errors.append(f"SeniorCitizen must be in {{0, 1}}, got: {senior_vals}")
                failed += 1
            else:
                passed += 1

        # 8. Service Relational Consistency Checks
        if "PhoneService" in df.columns and "MultipleLines" in df.columns:
            phone_no_inconsistent = int(
                ((df["PhoneService"] == "No") & (df["MultipleLines"] != "No phone service")).sum()
            )
            if phone_no_inconsistent > 0:
                warnings.append(
                    f"Found {phone_no_inconsistent} records where PhoneService='No' but MultipleLines is not 'No phone service'."
                )
            else:
                passed += 1

        if "InternetService" in df.columns:
            addons = [
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]
            internet_no_inconsistent = 0
            for addon in addons:
                if addon in df.columns:
                    internet_no_inconsistent += int(
                        ((df["InternetService"] == "No") & (df[addon] != "No internet service")).sum()
                    )
            if internet_no_inconsistent > 0:
                warnings.append(
                    f"Found {internet_no_inconsistent} inconsistencies where InternetService='No' but add-ons are not 'No internet service'."
                )
            else:
                passed += 1

        # 9. Pandera Schema Contract Check
        try:
            RAW_DATA_SCHEMA.validate(df, lazy=True)
            passed += 1
        except pandera.errors.SchemaErrors as exc:
            # Only treat as error if not the known whitespace TotalCharges issue
            schema_err_msgs = [f"{err.get('column', 'Schema')}: {err.get('error', 'Failed')}" for err in exc.failure_cases.to_dict(orient="records")[:5]]
            # If errors are purely due to TotalCharges whitespace, treat as warning
            only_total_charges = all("TotalCharges" in str(m) for m in schema_err_msgs)
            if only_total_charges:
                warnings.append(f"Pandera schema noted raw TotalCharges non-float strings (handled in cleaning step).")
                passed += 1
            else:
                errors.append(f"Pandera schema contract violations: {schema_err_msgs}")
                failed += 1

        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            checks_passed=passed,
            checks_failed=failed,
            metrics=metrics,
        )
        logger.info(result.summary())
        return result

    def validate_clean(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validates cleaned and standardized DataFrame before downstream modeling or SQL storage.
        """
        errors: List[str] = []
        warnings: List[str] = []
        passed = 0
        failed = 0

        # Null check
        null_sum = int(df.isnull().sum().sum())
        if null_sum > 0:
            errors.append(f"Cleaned dataset contains {null_sum} null values.")
            failed += 1
        else:
            passed += 1

        # Total charges type check
        if not np.issubdtype(df["total_charges"].dtype, np.floating):
            errors.append(f"total_charges is not floating point: {df['total_charges'].dtype}")
            failed += 1
        else:
            passed += 1

        # Churn label check
        if "churn_label" not in df.columns:
            errors.append("churn_label integer column missing from cleaned data.")
            failed += 1
        else:
            valid_labels = set(df["churn_label"].unique())
            if not valid_labels.issubset({0, 1}):
                errors.append(f"churn_label has invalid values: {valid_labels}")
                failed += 1
            else:
                passed += 1

        # Pandera Clean Schema check
        try:
            CLEAN_DATA_SCHEMA.validate(df, lazy=True)
            passed += 1
        except pandera.errors.SchemaErrors as exc:
            err_summary = [str(e) for e in exc.failure_cases.to_dict(orient="records")[:5]]
            errors.append(f"Clean schema violations: {err_summary}")
            failed += 1

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            checks_passed=passed,
            checks_failed=failed,
            metrics={"total_rows": len(df), "total_cols": len(df.columns)},
        )
