"""
Exploratory Data Analysis package for IBM Telco Customer Churn.
Provides statistical hypothesis testing, publication visualizations, and findings reporting.
"""

from src.eda.analyzer import CategoricalTestResult, EDAAnalyzer, NumericTestResult
from src.eda.report_generator import EDAReportGenerator
from src.eda.visualizer import EDAVisualizer

__all__ = [
    "EDAAnalyzer",
    "EDAVisualizer",
    "EDAReportGenerator",
    "NumericTestResult",
    "CategoricalTestResult",
]
