"""
Data ingestion module for IBM Telco Customer Churn dataset.
Handles fetching raw data from remote source, local caching, and loading into DataFrames.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Union
import requests
import pandas as pd

from src.config import BASE_DIR, get_settings


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def fetch_dataset(
    url: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    timeout: int = 30,
) -> Path:
    """
    Downloads the raw IBM Telco Customer Churn dataset from the canonical source URL.
    Saves to the configured raw data path.

    Args:
        url: Direct HTTP URL to raw CSV. Defaults to settings.data_raw_url.
        save_path: Local destination path. Defaults to settings.raw_data_file.
        force_download: If True, re-downloads even if the file exists locally.
        timeout: HTTP request timeout in seconds.

    Returns:
        Path to the saved raw CSV file.
    """
    settings = get_settings()
    source_url = url or settings.data_raw_url
    dest_path = Path(save_path) if save_path else settings.raw_data_file

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0 and not force_download:
        logger.info(
            f"Dataset already exists locally at {dest_path} ({dest_path.stat().st_size:,} bytes). Skipping download."
        )
        return dest_path

    logger.info(f"Downloading raw dataset from: {source_url}")
    try:
        response = requests.get(
            source_url,
            headers={"User-Agent": "TelcoChurnPlatform/1.0"},
            timeout=timeout,
            stream=True,
        )
        response.raise_for_status()

        total_bytes = 0
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)

        logger.info(f"Successfully downloaded {total_bytes:,} bytes to {dest_path}")
    except requests.RequestException as e:
        logger.error(f"Failed to download raw dataset from {source_url}: {e}")
        raise RuntimeError(f"Network error downloading Telco Churn dataset: {e}") from e

    return dest_path


def load_raw_data(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Loads raw CSV into a pandas DataFrame.
    Preserves exact string format for customerID and TotalCharges to allow explicit validation.

    Args:
        path: Path to the raw CSV file. Defaults to settings.raw_data_file.

    Returns:
        Raw pandas DataFrame.
    """
    settings = get_settings()
    file_path = Path(path) if path else settings.raw_data_file

    if not file_path.exists():
        logger.info(f"Raw data file not found at {file_path}. Fetching now...")
        fetch_dataset(save_path=file_path)

    logger.info(f"Loading raw dataset from {file_path}...")
    df = pd.read_csv(
        file_path,
        dtype={
            "customerID": str,
            "TotalCharges": str,
            "gender": str,
            "Partner": str,
            "Dependents": str,
            "PhoneService": str,
            "MultipleLines": str,
            "InternetService": str,
            "OnlineSecurity": str,
            "OnlineBackup": str,
            "DeviceProtection": str,
            "TechSupport": str,
            "StreamingTV": str,
            "StreamingMovies": str,
            "Contract": str,
            "PaperlessBilling": str,
            "PaymentMethod": str,
            "Churn": str,
        },
    )
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Ingest IBM Telco Customer Churn dataset.")
    parser.add_argument("--url", type=str, help="Source URL to download CSV from.")
    parser.add_argument("--output", type=str, help="Destination file path.")
    parser.add_argument("--force", action="store_true", help="Force redownload even if exists.")
    args = parser.parse_args()

    saved_file = fetch_dataset(url=args.url, save_path=args.output, force_download=args.force)
    df = load_raw_data(saved_file)
    print(f"Dataset successfully ingested: {saved_file} with shape {df.shape}")


if __name__ == "__main__":
    main()
