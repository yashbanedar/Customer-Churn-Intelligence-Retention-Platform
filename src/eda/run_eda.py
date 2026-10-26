"""
CLI Runner and Pipeline Orchestrator for Phase 3 Exploratory Data Analysis (EDA).
Executes statistical hypothesis testing, renders publication-grade charts,
and compiles the comprehensive findings report.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Union

from src.config import BASE_DIR, get_settings
from src.data.cleaner import clean_dataset
from src.data.ingest import load_raw_data
from src.eda.analyzer import EDAAnalyzer
from src.eda.report_generator import EDAReportGenerator
from src.eda.visualizer import EDAVisualizer


logger = logging.getLogger("src.eda.run_eda")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_eda(
    data_path: Optional[Union[str, Path]] = None,
    output_figures_dir: Optional[Union[str, Path]] = None,
    output_metrics_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """
    Executes complete Phase 3 EDA:
    1. Loads validated dataset.
    2. Runs statistical analyses (target balance, Mann-Whitney U, Chi-Square, Cramér's V, leakage audit).
    3. Renders 7 reproducible static publication charts + 1 interactive Plotly HTML.
    4. Compiles formal findings report into Markdown and JSON.

    Returns:
        True if EDA finished successfully, False otherwise.
    """
    settings = get_settings()
    logger.info("=" * 70)
    logger.info("STARTING PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("=" * 70)

    # 1. Load data
    clean_parquet = settings.processed_data_file
    if clean_parquet.exists() and not data_path:
        logger.info(f"Loading pre-validated clean dataset from: {clean_parquet}")
        df = clean_dataset(load_raw_data())
    else:
        logger.info("Loading and cleaning raw dataset...")
        df = clean_dataset(load_raw_data(data_path))

    logger.info(f"Dataset successfully loaded. Records: {len(df):,}, Features: {len(df.columns)}")

    # 2. Statistical Analysis
    logger.info("--> STEP 1: Running Statistical Hypothesis Tests & Association Analyses...")
    analyzer = EDAAnalyzer(df=df)
    target_stats = analyzer.analyze_target_balance()
    logger.info(f"Target Balance: {target_stats['churn_rate_pct']}% Churn ({target_stats['churned_count']:,} / {target_stats['total_customers']:,})")

    # 3. Chart Generation
    logger.info("--> STEP 2: Rendering Publication-Grade Figures & Interactive Explorer...")
    fig_dir = Path(output_figures_dir) if output_figures_dir else BASE_DIR / "artifacts" / "figures"
    visualizer = EDAVisualizer(df=df, output_dir=fig_dir)
    rendered_figs = visualizer.render_all()
    logger.info(f"✅ Successfully rendered {len(rendered_figs)} figures to {fig_dir}:")
    for name, path in rendered_figs.items():
        logger.info(f"   - {name}: {path.name} ({path.stat().st_size:,} bytes)")

    # 4. Findings Report Generation
    logger.info("--> STEP 3: Compiling Formal Findings Report & Statistical Tables...")
    metrics_dir = Path(output_metrics_dir) if output_metrics_dir else BASE_DIR / settings.metrics_artifacts_dir
    report_gen = EDAReportGenerator(analyzer=analyzer, output_dir=metrics_dir)
    report_data = report_gen.generate_report()
    logger.info(f"✅ Findings report generated: {metrics_dir / 'eda_findings_report.md'}")
    logger.info(f"✅ Summary JSON generated: {metrics_dir / 'eda_summary.json'}")

    logger.info("=" * 70)
    logger.info("PHASE 3 EDA COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run Exploratory Data Analysis for Telco Customer Churn.")
    parser.add_argument("--data-path", type=str, help="Custom path to dataset CSV or Parquet.")
    parser.add_argument("--figures-dir", type=str, help="Custom output directory for figures.")
    parser.add_argument("--metrics-dir", type=str, help="Custom output directory for reports.")
    args = parser.parse_args()

    success = run_eda(
        data_path=args.data_path,
        output_figures_dir=args.figures_dir,
        output_metrics_dir=args.metrics_dir,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
