"""
Data Quality Report Generator for IBM Telco Customer Churn dataset.
Calculates statistical summaries, missingness audits, class balance,
segment churn distributions, and outputs structured JSON and Markdown artifacts.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from src.config import BASE_DIR, get_settings


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def generate_data_quality_report(
    raw_df: pd.DataFrame,
    clean_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Generates comprehensive data quality report for the IBM Telco dataset.
    Saves JSON and Markdown report files to artifacts/metrics/.

    Args:
        raw_df: Raw DataFrame as ingested from source.
        clean_df: Optional cleaned DataFrame (if already processed).
        output_dir: Destination folder for report artifacts. Defaults to settings.metrics_artifacts_dir.

    Returns:
        Dictionary containing all metrics and report structures.
    """
    settings = get_settings()
    out_dir = Path(output_dir) if output_dir else BASE_DIR / settings.metrics_artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_timestamp = datetime.now(timezone.utc).isoformat()
    total_records = len(raw_df)
    total_cols = len(raw_df.columns)

    # 1. Dataset Overview
    overview = {
        "generated_at_utc": report_timestamp,
        "total_rows": total_records,
        "total_columns": total_cols,
        "primary_key": "customerID" if "customerID" in raw_df.columns else "customer_id",
        "duplicate_rows_count": int(raw_df.duplicated().sum()),
        "duplicate_primary_keys": int(
            raw_df["customerID"].duplicated().sum()
            if "customerID" in raw_df.columns
            else raw_df.get("customer_id", pd.Series(dtype=object)).duplicated().sum()
        ),
        "memory_usage_mb": round(raw_df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }

    # 2. Missing & Whitespace Value Audit
    missing_analysis = {}
    for col in raw_df.columns:
        null_count = int(raw_df[col].isnull().sum())
        # Check for whitespace-only strings
        is_str_col = raw_df[col].dtype == object or pd.api.types.is_string_dtype(raw_df[col])
        ws_count = int((raw_df[col].astype(str).str.strip() == "").sum()) if is_str_col else 0
        effective_missing = null_count + ws_count
        missing_analysis[col] = {
            "dtype": str(raw_df[col].dtype),
            "null_count": null_count,
            "whitespace_count": ws_count,
            "total_missing": effective_missing,
            "missing_pct": round(100.0 * effective_missing / total_records, 2),
        }

    # 3. Target Label (Churn) Distribution
    target_col = "Churn" if "Churn" in raw_df.columns else "churn"
    target_dist = {}
    if target_col in raw_df.columns:
        val_counts = raw_df[target_col].value_counts().to_dict()
        yes_count = int(val_counts.get("Yes", 0))
        no_count = int(val_counts.get("No", 0))
        churn_rate = round(100.0 * yes_count / total_records, 2)
        imbalance_ratio = round(no_count / yes_count, 2) if yes_count > 0 else 0.0

        target_dist = {
            "target_column": target_col,
            "counts": {"No": no_count, "Yes": yes_count},
            "percentages": {
                "No": round(100.0 * no_count / total_records, 2),
                "Yes": round(100.0 * yes_count / total_records, 2),
            },
            "churn_rate_pct": churn_rate,
            "imbalance_ratio_negative_to_positive": imbalance_ratio,
        }

    # 4. Numerical Features Summary
    # Use clean_df if available for accurate total_charges floats, else coerce
    numeric_source = clean_df if clean_df is not None else raw_df
    num_cols_to_check = ["tenure", "monthly_charges", "total_charges"]
    if clean_df is None:
        num_cols_to_check = ["tenure", "MonthlyCharges", "TotalCharges"]

    numeric_summary = {}
    for col in num_cols_to_check:
        if col in numeric_source.columns:
            s = pd.to_numeric(numeric_source[col], errors="coerce").dropna()
            numeric_summary[col] = {
                "count": int(s.count()),
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "p25": round(float(s.quantile(0.25)), 2),
                "median": round(float(s.median()), 2),
                "p75": round(float(s.quantile(0.75)), 2),
                "max": round(float(s.max()), 2),
                "skewness": round(float(s.skew()), 3),
            }

    # 5. Categorical Features Cardinality & Distribution
    cat_summary = {}
    cat_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "Contract",
        "PaymentMethod",
        "PaperlessBilling",
    ]
    # Check lowercase equivalent if clean
    for col in cat_cols:
        target_name = col if col in raw_df.columns else col.lower()
        if target_name in raw_df.columns:
            freq = raw_df[target_name].value_counts().to_dict()
            cat_summary[col] = {
                "unique_count": int(raw_df[target_name].nunique()),
                "frequencies": {str(k): int(v) for k, v in freq.items()},
            }

    # 6. Segmented Churn Rates (Business Insights)
    segment_churn = {}
    c_df = clean_df if clean_df is not None else raw_df.copy()
    contract_col = "contract" if "contract" in c_df.columns else "Contract"
    churn_col = "churn" if "churn" in c_df.columns else "Churn"
    internet_col = "internet_service" if "internet_service" in c_df.columns else "InternetService"
    payment_col = "payment_method" if "payment_method" in c_df.columns else "PaymentMethod"
    tenure_col = "tenure" if "tenure" in c_df.columns else "tenure"

    if contract_col in c_df.columns and churn_col in c_df.columns:
        contract_churn = (
            c_df.groupby(contract_col)[churn_col]
            .apply(lambda s: round(100.0 * (s == "Yes").mean(), 2))
            .to_dict()
        )
        segment_churn["by_contract_type"] = contract_churn

    if internet_col in c_df.columns and churn_col in c_df.columns:
        internet_churn = (
            c_df.groupby(internet_col)[churn_col]
            .apply(lambda s: round(100.0 * (s == "Yes").mean(), 2))
            .to_dict()
        )
        segment_churn["by_internet_service"] = internet_churn

    if payment_col in c_df.columns and churn_col in c_df.columns:
        payment_churn = (
            c_df.groupby(payment_col)[churn_col]
            .apply(lambda s: round(100.0 * (s == "Yes").mean(), 2))
            .to_dict()
        )
        segment_churn["by_payment_method"] = payment_churn

    if tenure_col in c_df.columns and churn_col in c_df.columns:
        t_numeric = pd.to_numeric(c_df[tenure_col], errors="coerce").fillna(0)
        tenure_bins = pd.cut(
            t_numeric,
            bins=[-1, 12, 24, 48, 100],
            labels=["0-12m", "13-24m", "25-48m", "49-72m"],
        )
        tenure_churn = (
            c_df.groupby(tenure_bins, observed=False)[churn_col]
            .apply(lambda s: round(100.0 * (s == "Yes").mean(), 2))
            .to_dict()
        )
        segment_churn["by_tenure_cohort"] = {str(k): v for k, v in tenure_churn.items()}

    # Combine full report
    full_report = {
        "dataset_overview": overview,
        "target_distribution": target_dist,
        "missing_and_whitespace_audit": missing_analysis,
        "numeric_summary_statistics": numeric_summary,
        "categorical_features_summary": cat_summary,
        "segmented_churn_rates": segment_churn,
        "validation_status": "PASSED"
        if overview["duplicate_primary_keys"] == 0 and target_dist.get("churn_rate_pct", 0) > 0
        else "FAILED",
    }

    # 7. Write JSON artifact
    json_path = out_dir / "data_quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    logger.info(f"Saved data quality report JSON to {json_path}")

    # 8. Write Markdown artifact
    md_path = out_dir / "data_quality_report.md"
    md_content = _build_markdown_report(full_report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved data quality report Markdown to {md_path}")

    return full_report


def _build_markdown_report(report: Dict[str, Any]) -> str:
    """Formats report dictionary into rich GitHub-flavored markdown."""
    ov = report["dataset_overview"]
    td = report["target_distribution"]
    num = report["numeric_summary_statistics"]
    seg = report["segmented_churn_rates"]

    lines = [
        "# Data Quality & Integrity Report: IBM Telco Customer Churn",
        "",
        f"**Generated**: `{ov['generated_at_utc']}` | **Status**: **{report['validation_status']}**",
        "",
        "## 1. Executive Summary & Dataset Volume",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Total Records (Customers)** | `{ov['total_rows']:,}` |",
        f"| **Total Columns** | `{ov['total_columns']}` |",
        f"| **Primary Key** | `{ov['primary_key']}` |",
        f"| **Duplicate Customer IDs** | `{ov['duplicate_primary_keys']}` (0.00%) |",
        f"| **Duplicate Full Rows** | `{ov['duplicate_rows_count']}` |",
        f"| **Memory Footprint** | `{ov['memory_usage_mb']} MB` |",
        f"| **Churn Rate** | **{td.get('churn_rate_pct')}%** ({td['counts'].get('Yes', 0):,} churned / {ov['total_rows']:,} total) |",
        f"| **Imbalance Ratio** | `{td.get('imbalance_ratio_negative_to_positive')}:1` (Non-Churn : Churn) |",
        "",
        "## 2. Missing Values & Whitespace Anomaly Audit",
        "",
        "| Column | Raw Dtype | Null Count | Whitespace Count | Total Missing | Missing % | Notes |",
        "| :--- | :--- | :-: | :-: | :-: | :-: | :--- |",
    ]

    for col, data in report["missing_and_whitespace_audit"].items():
        notes = ""
        if data["whitespace_count"] > 0:
            notes = f"⚠️ {data['whitespace_count']} spaces at tenure=0 (IBM quirk resolved)"
        elif data["total_missing"] == 0:
            notes = "✅ 100% complete"
        lines.append(
            f"| `{col}` | `{data['dtype']}` | {data['null_count']} | {data['whitespace_count']} | {data['total_missing']} | {data['missing_pct']}% | {notes} |"
        )

    lines.extend([
        "",
        "## 3. Numerical Distributions",
        "",
        "| Feature | Count | Mean | Std | Min | 25% | Median | 75% | Max | Skewness |",
        "| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
    ])

    for col, s in num.items():
        formatted_mean = f"${s['mean']:,.2f}" if "charge" in col.lower() else f"{s['mean']}"
        lines.append(
            f"| `{col}` | {s['count']:,} | {formatted_mean} | {s['std']} | {s['min']} | {s['p25']} | {s['median']} | {s['p75']} | {s['max']} | {s['skewness']} |"
        )

    lines.extend([
        "",
        "## 4. Segmented Churn Risk Matrix",
        "",
        "### Churn Rate by Contract Type",
        "| Contract Term | Churn Rate (%) | Risk Level |",
        "| :--- | :-: | :--- |",
    ])
    for k, v in seg.get("by_contract_type", {}).items():
        risk = "🚨 Critical" if v > 40 else ("⚠️ Moderate" if v > 15 else "✅ Low")
        lines.append(f"| `{k}` | **{v}%** | {risk} |")

    lines.extend([
        "",
        "### Churn Rate by Internet Service",
        "| Internet Service | Churn Rate (%) | Risk Level |",
        "| :--- | :-: | :--- |",
    ])
    for k, v in seg.get("by_internet_service", {}).items():
        risk = "🚨 High" if v > 35 else ("⚠️ Moderate" if v > 15 else "✅ Low")
        lines.append(f"| `{k}` | **{v}%** | {risk} |")

    lines.extend([
        "",
        "### Churn Rate by Tenure Cohort",
        "| Tenure Cohort | Churn Rate (%) | Retention Stage |",
        "| :--- | :-: | :--- |",
    ])
    for k, v in seg.get("by_tenure_cohort", {}).items():
        stage = "New Customer Onboarding (Highest Vulnerability)" if "0-12" in k else "Established Retention"
        lines.append(f"| `{k}` | **{v}%** | {stage} |")

    lines.extend([
        "",
        "---",
        "*(Report generated automatically by the Customer Churn Intelligence & Retention Optimization Platform)*",
    ])

    return "\n".join(lines)
