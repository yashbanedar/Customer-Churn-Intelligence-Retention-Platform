"""
Unit tests for database loading and schema integrity.
"""

from pathlib import Path
import pytest
from sqlalchemy import create_engine, text

from src.data.cleaner import clean_dataset
from src.data.db_loader import execute_schema_ddl, load_dataset_to_sql
from src.data.ingest import load_raw_data


@pytest.fixture
def sqlite_test_engine(tmp_path):
    """Provides a fresh isolated SQLite database engine for testing."""
    db_file = tmp_path / "test_telco.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}")
    return engine


def test_schema_ddl_and_db_loader(sqlite_test_engine):
    """
    Verifies that execute_schema_ddl creates tables and load_dataset_to_sql populates all 3 normalized tables.
    """
    raw_df = load_raw_data()
    sample_clean = clean_dataset(raw_df.head(100))

    counts = load_dataset_to_sql(sample_clean, engine=sqlite_test_engine, recreate_tables=True)

    assert counts["customers"] == 100
    assert counts["customer_services"] == 100
    assert counts["customer_contracts"] == 100

    # Query tables to verify foreign key relationship and schema consistency
    with sqlite_test_engine.connect() as conn:
        res = conn.execute(
            text("""
            SELECT c.customer_id, c.gender, s.internet_service, co.monthly_charges, co.churn
            FROM customers c
            JOIN customer_services s ON c.customer_id = s.customer_id
            JOIN customer_contracts co ON c.customer_id = co.customer_id
            LIMIT 5
            """)
        ).fetchall()

        assert len(res) == 5
        for row in res:
            assert len(row[0]) > 0  # customer_id exists
            assert row[1] in ("Male", "Female")  # gender valid
            assert row[4] in ("Yes", "No")  # churn valid
