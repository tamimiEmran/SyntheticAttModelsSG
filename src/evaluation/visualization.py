# src/evaluation/visualization.py
"""
Functions for visualizing model evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple

# --- Plotting Defaults (Consider moving to a central config or utils) ---
def set_plotting_defaults(figsize=(10, 6), dpi=100, font_size=12, **kwargs):
    """Applies some default matplotlib settings for consistency."""
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['axes.titlesize'] = font_size + 2
    plt.rcParams['xtick.labelsize'] = font_size - 1
    plt.rcParams['ytick.labelsize'] = font_size - 1
    plt.rcParams['legend.fontsize'] = font_size - 1
    # Apply any other custom defaults
    plt.rcParams.update(kwargs)

set_plotting_defaults() # Apply defaults when module is loaded

# --- Plotting Functions ---

def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: Optional[float] = None,
    model_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes:
    """
    Plots a single ROC curve.

    Args:
        fpr (np.ndarray): False Positive Rates.
        tpr (np.ndarray): True Positive Rates.
        roc_auc (Optional[float]): Area under the curve score to display.
        model_name (Optional[str]): Name of the model for the legend.
        ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates a new figure.
        **plot_kwargs: Additional keyword arguments passed to plt.plot.

    Returns:
        plt.Axes: The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    label = model_name if model_name else ""
    if roc_auc is not None:
        label += f' (AUC = {roc_auc:.3f})'

    ax.plot(fpr, tpr, label=label.strip(), **plot_kwargs)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6) # Diagonal reference line
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return ax

def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: Optional[float] = None,
    model_name: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes:
    """
    Plots a single Precision-Recall curve.

    Args:
        precision (np.ndarray): Precision values.
        recall (np.ndarray): Recall values.
        pr_auc (Optional[float]): Area under the curve score to display.
        model_name (Optional[str]): Name of the model for the legend.
        ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates a new figure.
        **plot_kwargs: Additional keyword arguments passed to plt.plot.

    Returns:
        plt.Axes: The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    label = model_name if model_name else ""
    if pr_auc is not None:
        label += f' (AUC = {pr_auc:.3f})'

    # Note: Plotting recall on x-axis, precision on y-axis
    ax.plot(recall, precision, label=label.strip(), **plot_kwargs)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left") # Often better placement for PR curves
    ax.grid(alpha=0.3)
    return ax

def plot_comparison_bar(
    results_dict: Dict[str, Dict[str, List[float]]],
    metric: str = 'ROC-AUC',
    title: Optional[str] = None,
    sort_by_metric: bool = True,
    ax: Optional[plt.Axes] = None,
    bar_width: float = 0.35,
    capsize: int = 5
) -> plt.Axes:
    """
    Creates a bar plot comparing a specific metric across different models,
    with error bars representing standard deviation across folds/runs.

    Args:
        results_dict (Dict[str, Dict[str, List[float]]]): Dictionary where keys are
            model names and values are dictionaries of metrics, each containing
            a list of results (e.g., from different folds).
            Example: {'ModelA': {'ROC-AUC': [0.8, 0.85], 'F1': [0.7, 0.75]}, ...}
        metric (str): The metric to plot (must be a key in the inner dictionaries).
        title (Optional[str]): Title for the plot. Defaults to 'Model Comparison'.
        sort_by_metric (bool): If True, sort models by the mean of the chosen metric.
        ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates a new figure.
        bar_width (float): Width of the bars.
        capsize (int): Size of the error bar caps.

    Returns:
        plt.Axes: The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    model_names = list(results_dict.keys())
    if not model_names:
        print("Warning: results_dict is empty. Cannot generate plot.")
        return ax

    means = []
    stds = []
    valid_model_names = []

    for name in model_names:
        if metric in results_dict[name]:
            values = results_dict[name][metric]
            if values: # Check if list is not empty
                means.append(np.mean(values))
                stds.append(np.std(values))
                valid_model_names.append(name)
            else:
                print(f"Warning: Empty list for metric '{metric}' in model '{name}'. Skipping.")
        else:
            print(f"Warning: Metric '{metric}' not found for model '{name}'. Skipping.")

    if not valid_model_names:
        print(f"Warning: No valid data found for metric '{metric}'. Cannot generate plot.")
        return ax

    means = np.array(means)
    stds = np.array(stds)
    valid_model_names = np.array(valid_model_names)

    if sort_by_metric:
        order = means.argsort()[::-1] # Sort descending
        means = means[order]
        stds = stds[order]
        valid_model_names = valid_model_names[order]

    x = np.arange(len(valid_model_names))

    ax.bar(x, means, yerr=stds, width=bar_width, capsize=capsize, label=metric)

    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_model_names, rotation=45, ha="right")
    ax.set_title(title if title else f'Model Comparison - {metric}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout() # Adjust layout

    return ax


def plot_metric_bars(
    results_dict: Dict[str, Dict[str, List[float]]],
    metrics_to_plot: List[str] = ['Precision', 'Recall', 'F1'],
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    bar_width_factor: float = 0.8 # How much of the available space the group of bars should take
) -> plt.Axes:
    """
    Creates a grouped bar plot comparing multiple metrics (e.g., Precision, Recall, F1)
    for each model. Error bars show standard deviation across folds.

    Args:
        results_dict (Dict[str, Dict[str, List[float]]]): Results dictionary (same format as plot_comparison_bar).
        metrics_to_plot (List[str]): List of metric names to include in the grouped plot.
        title (Optional[str]): Title for the plot. Defaults to 'Model Performance Comparison'.
        ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates a new figure.
        bar_width_factor (float): Factor to control the width of the group of bars per model.

    Returns:
        plt.Axes: The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    model_names = list(results_dict.keys())
    if not model_names:
        print("Warning: results_dict is empty.")
        return ax

    n_metrics = len(metrics_to_plot)
    n_models = len(model_names)
    total_bar_width = bar_width_factor / n_metrics # Width of a single bar
    x = np.arange(n_models) # Base positions for model groups

    for i, metric in enumerate(metrics_to_plot):
        means = []
        stds = []
        current_model_names = [] # Track models that have this metric

        for name in model_names:
            if metric in results_dict[name] and results_dict[name][metric]:
                values = results_dict[name][metric]
                means.append(np.mean(values))
                stds.append(np.std(values))
                current_model_names.append(name) # Only include if metric exists
            # else: Add placeholder if necessary for alignment? Maybe better to filter models

        if not means: # Skip if no model has this metric
             print(f"Warning: No data found for metric '{metric}'. Skipping.")
             continue

        # Calculate bar positions for this metric
        bar_positions = x[:len(means)] + (i - (n_metrics - 1) / 2) * total_bar_width

        ax.bar(bar_positions, means, yerr=stds, width=total_bar_width, label=metric, capsize=3)

    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right") # Use all original model names for ticks
    ax.set_title(title if title else 'Model Performance Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return ax


def plot_metric_trends_with_errorbars(
    data_list: List[List[float]],
    x_labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    ax: Optional[plt.Axes] = None,
    **plot_kwargs
) -> plt.Axes:
    """
    Plots a trend line with error bars, often used for showing performance
    during iterative processes (like attack removal).

    Args:
        data_list (List[List[float]]): A list where each inner list contains
                                        results (e.g., AUCs across folds) for one step/iteration.
        x_labels (List[str]): Labels for the x-axis corresponding to each step/iteration.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates a new figure.
        **plot_kwargs: Additional keyword arguments passed to plt.errorbar.

    Returns:
        plt.Axes: The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if len(data_list) != len(x_labels):
        raise ValueError("Length of data_list must match length of x_labels.")

    means = [np.mean(inner_list) for inner_list in data_list]
    stds = [np.std(inner_list) for inner_list in data_list]
    x_values = np.arange(len(x_labels))

    ax.errorbar(x_values, means, yerr=stds, fmt='-o', capsize=5, capthick=1.5, **plot_kwargs)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return ax
