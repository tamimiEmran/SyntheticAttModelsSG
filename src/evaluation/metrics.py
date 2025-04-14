# src/evaluation/metrics.py
"""
Functions for calculating model evaluation metrics.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    classification_report
)
from typing import Dict, Any, Tuple, Optional

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    positive_label: int = 1
) -> Dict[str, float]:
    """
    Calculates standard binary classification metrics.

    Args:
        y_true (np.ndarray): Ground truth labels (0 or 1).
        y_pred (np.ndarray): Predicted labels (0 or 1).
        y_pred_proba (Optional[np.ndarray]): Predicted probabilities for the
                                             positive class. Required for AUC.
        positive_label (int): The label considered as positive (default: 1).

    Returns:
        Dict[str, float]: A dictionary containing calculated metrics:
                          'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'.
                          'ROC-AUC' will be None if y_pred_proba is not provided.
    """
    metrics = {}

    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Basic metrics
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)

    # ROC AUC
    if y_pred_proba is not None:
        y_pred_proba = np.asarray(y_pred_proba)
        try:
            # Ensure y_pred_proba corresponds to the positive class
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                 # Assume probabilities for class 0 and 1 are given
                 positive_class_proba = y_pred_proba[:, positive_label]
            elif y_pred_proba.ndim == 1:
                 # Assume probabilities for the positive class are given directly
                 positive_class_proba = y_pred_proba
            else:
                 raise ValueError("y_pred_proba has unexpected shape.")

            metrics['ROC-AUC'] = roc_auc_score(y_true, positive_class_proba)
        except ValueError as e:
            # Handle cases like only one class present in y_true
            print(f"Warning: Could not calculate ROC-AUC. Error: {e}")
            metrics['ROC-AUC'] = None
        except Exception as e:
            print(f"An unexpected error occurred during ROC-AUC calculation: {e}")
            metrics['ROC-AUC'] = None
    else:
        metrics['ROC-AUC'] = None
        print("Warning: y_pred_proba not provided, ROC-AUC cannot be calculated.")

    # Clean up None values before returning if desired, or keep them
    # metrics = {k: v for k, v in metrics.items() if v is not None}

    return metrics


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **kwargs
) -> Dict[str, Any]:
    """
    Generates a classification report as a dictionary.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        **kwargs: Additional arguments passed to sklearn.metrics.classification_report.

    Returns:
        Dict[str, Any]: The classification report as a dictionary.
    """
    return classification_report(y_true, y_pred, output_dict=True, **kwargs)


def get_roc_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    positive_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculates data needed for plotting an ROC curve.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
        positive_label (int): The label considered as positive.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: fpr, tpr, roc_auc score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba, pos_label=positive_label)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def get_pr_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    positive_label: int = 1
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculates data needed for plotting a Precision-Recall curve.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.
        positive_label (int): The label considered as positive.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: precision, recall, pr_auc score.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba, pos_label=positive_label)
    pr_auc = auc(recall, precision) # Note: x=recall, y=precision for PR AUC
    return precision, recall, pr_auc

def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> np.ndarray:
    """
    Calculates the confusion matrix.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        labels (Optional[list]): List of labels to index the matrix.

    Returns:
        np.ndarray: The confusion matrix.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)
