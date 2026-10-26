"""
Database Loader Module for Customer Churn Platform.
Populates normalized relational tables in PostgreSQL (or SQLite fallback)
from validated and cleaned Telco Churn datasets.
Keeps all connection credentials and secrets strictly in environment variables.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from src.config import BASE_DIR, get_settings


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def get_db_engine(db_url: Optional[str] = None) -> Engine:
    """
    Creates and returns a SQLAlchemy Engine using environment configuration.
    Credentials are read securely via Pydantic settings.
    """
    settings = get_settings()
    url = db_url or settings.database_url
    # Hide password in logs
    safe_log_url = url.split("@")[-1] if "@" in url else url
    logger.info(f"Connecting to database target: {safe_log_url}")
    return create_engine(url)


def execute_schema_ddl(engine: Engine, schema_file: Optional[Union[str, Path]] = None) -> None:
    """
    Executes DDL statements from sql/schema.sql on the database.
    Handles compatibility between PostgreSQL and SQLite.
    """
    schema_path = Path(schema_file) if schema_file else BASE_DIR / "sql" / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema DDL file not found at: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        ddl = f.read()

    # If running against SQLite, adjust PostgreSQL-specific keywords
    is_sqlite = engine.dialect.name == "sqlite"
    if is_sqlite:
        # SQLite doesn't support DROP TABLE ... CASCADE
        ddl = ddl.replace("CASCADE;", ";")
        # Replace SERIAL with INTEGER PRIMARY KEY AUTOINCREMENT
        ddl = ddl.replace("prediction_id SERIAL PRIMARY KEY,", "prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,")

    # Split into individual executable statements
    statements = [stmt.strip() for stmt in ddl.split(";") if stmt.strip()]

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))

    logger.info(f"Executed {len(statements)} DDL schema statements on {engine.dialect.name}.")


def load_dataset_to_sql(
    df: pd.DataFrame,
    engine: Optional[Engine] = None,
    recreate_tables: bool = True,
) -> Dict[str, int]:
    """
    Splits cleaned Telco dataset into normalized 3NF relational tables and loads into database.

    Normalized Tables:
    1. customers (demographics)
    2. customer_services (telecom and internet subscriptions)
    3. customer_contracts (contract terms, monthly/total charges, churn)

    Args:
        df: Cleaned Telco DataFrame.
        engine: Optional SQLAlchemy engine. Defaults to engine configured in settings.
        recreate_tables: If True, executes schema.sql DDL before loading.

    Returns:
        Dictionary with record counts loaded into each table.
    """
    active_engine = engine or get_db_engine()

    if recreate_tables:
        execute_schema_ddl(active_engine)

    # 1. Prepare customers table
    customers_cols = ["customer_id", "gender", "senior_citizen", "partner", "dependents"]
    customers_df = df[customers_cols].drop_duplicates(subset=["customer_id"])

    # 2. Prepare customer_services table
    services_cols = [
        "customer_id",
        "phone_service",
        "multiple_lines",
        "internet_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]
    services_df = df[services_cols].drop_duplicates(subset=["customer_id"])

    # 3. Prepare customer_contracts table
    contracts_cols = [
        "customer_id",
        "tenure",
        "contract",
        "paperless_billing",
        "payment_method",
        "monthly_charges",
        "total_charges",
        "churn",
    ]
    contracts_df = df[contracts_cols].drop_duplicates(subset=["customer_id"])

    # Bulk insert into tables in relational order (parent first)
    with active_engine.begin() as conn:
        logger.info(f"Writing {len(customers_df):,} records into 'customers'...")
        customers_df.to_sql("customers", conn, if_exists="append", index=False, chunksize=1000)

        logger.info(f"Writing {len(services_df):,} records into 'customer_services'...")
        services_df.to_sql("customer_services", conn, if_exists="append", index=False, chunksize=1000)

        logger.info(f"Writing {len(contracts_df):,} records into 'customer_contracts'...")
        contracts_df.to_sql("customer_contracts", conn, if_exists="append", index=False, chunksize=1000)

    # Verify counts
    with active_engine.connect() as conn:
        c_count = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar()
        s_count = conn.execute(text("SELECT COUNT(*) FROM customer_services")).scalar()
        k_count = conn.execute(text("SELECT COUNT(*) FROM customer_contracts")).scalar()

    results = {
        "customers": int(c_count or 0),
        "customer_services": int(s_count or 0),
        "customer_contracts": int(k_count or 0),
    }

    logger.info(f"Database ingestion verified: {results}")
    return results
