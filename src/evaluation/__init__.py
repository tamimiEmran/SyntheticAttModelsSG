# src/evaluation/__init__.py
"""
Evaluation Module

Provides functions for calculating performance metrics and visualizing results.
"""
from .metrics import calculate_metrics, classification_report_dict
from .visualization import (
    plot_comparison_bar,
    plot_roc_curve, # Added for completeness
    plot_precision_recall_curve, # Added for completeness
    plot_metric_bars,
    plot_metric_trends_with_errorbars
)

__all__ = [
    "calculate_metrics",
    "classification_report_dict",
    "plot_comparison_bar",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_metric_bars",
    "plot_metric_trends_with_errorbars"
]
