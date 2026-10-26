# Data Directory Architecture

This directory houses the raw and processed data assets for the **Customer Churn Intelligence & Retention Optimization Platform**.

## Directory Structure

```
data/
├── raw/
│   └── Telco-Customer-Churn.csv         # Original unaltered raw dataset from IBM source
├── processed/
│   ├── telco_churn_clean.parquet        # High-performance Parquet format with strict typing
│   └── telco_churn_clean.csv            # Cleaned CSV with standardized snake_case schema
└── telco_churn.db                       # Local SQLite relational database (fallback/testing)
```

## Data Schema & Lineage Reference
Detailed field definitions, value domains, and data quality specifications are documented in:
[docs/DATA_SOURCE_AND_SCHEMA.md](../docs/DATA_SOURCE_AND_SCHEMA.md).

## Running Ingestion & Validation
To download, validate, clean, and generate a quality report, run:
```bash
python -m src.data.pipeline --download --validate --clean --report --load-db
```
