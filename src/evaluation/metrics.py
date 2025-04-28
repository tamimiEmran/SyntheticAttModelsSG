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

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, brier_score_loss
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, brier_score_loss
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any


@dataclass
class ConfusionMetrics:
    """Metrics derived from the confusion matrix"""
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    def as_dict(self) -> Dict[str, int]:
        """Convert to dictionary representation"""
        return {
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        }


@dataclass
class DatasetMetrics:
    """Metrics for a single dataset (train, validation, or test)"""
    # Probability metrics
    log_loss: float
    brier_score: float
    
    # Curve metrics
    auc_roc: float
    auc_pr: float
    
    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    specificity: float = 0.0
    sensitivity: float = 0.0
    balanced_accuracy: float = 0.0
    positive_predictive_value: float = 0.0
    negative_predictive_value: float = 0.0
    
    # Confusion matrix
    confusion: Optional[ConfusionMetrics] = None
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_name == 'confusion' and field_value is not None:
                result['confusion_matrix'] = field_value.as_dict()
            else:
                result[field_name] = field_value
        return result


@dataclass
class EvaluationResults:
    """Complete evaluation results for all datasets"""
    train: DatasetMetrics
    validation: DatasetMetrics
    threshold: float
    test: Optional[DatasetMetrics] = None
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = {
            'train': self.train.as_dict(),
            'val': self.validation.as_dict(),
            'threshold': self.threshold
        }
        if self.test is not None:
            result['test'] = self.test.as_dict()
        return result


class Evaluator:
    """Class for evaluating binary classification models"""
    
    def __init__(self, 
                 find_optimal_threshold: bool = True, 
                 plot_curves: bool = True):
        """
        Initialize the evaluator
        
        Parameters:
        -----------
        find_optimal_threshold : bool, default=True
            Whether to find optimal threshold on validation data
        plot_curves : bool, default=True
            Whether to plot ROC and PR curves
        """
        self.find_optimal_threshold = find_optimal_threshold
        self.plot_curves = plot_curves
    
    def evaluate(self, 
                y_train_pred: np.ndarray, 
                y_train_true: np.ndarray,
                y_val_pred: np.ndarray, 
                y_val_true: np.ndarray,
                y_test_pred: Optional[np.ndarray] = None, 
                y_test_true: Optional[np.ndarray] = None,
                custom_threshold: Optional[float] = None) -> EvaluationResults:
        """
        Evaluate binary classification results
        
        Parameters:
        -----------
        y_train_pred : array-like
            Predicted probabilities for training data
        y_train_true : array-like
            True binary labels for training data
        y_val_pred : array-like
            Predicted probabilities for validation data
        y_val_true : array-like
            True binary labels for validation data
        y_test_pred : array-like, optional
            Predicted probabilities for test data
        y_test_true : array-like, optional
            True binary labels for test data
        custom_threshold : float, optional
            Custom threshold to use instead of finding optimal threshold
            
        Returns:
        --------
        results : EvaluationResults
            Complete evaluation metrics for all datasets
        """
        # Convert inputs to numpy arrays
        y_train_pred = np.array(y_train_pred)
        y_train_true = np.array(y_train_true)
        y_val_pred = np.array(y_val_pred)
        y_val_true = np.array(y_val_true)
        
        # if predicitions are 2d then covert to 1d
        if y_train_pred.ndim == 2:
            y_train_pred = y_train_pred[:, 1]
        if y_val_pred.ndim == 2:
            y_val_pred = y_val_pred[:, 1]
        if y_test_pred is not None and y_test_pred.ndim == 2:
            y_test_pred = y_test_pred[:, 1]

        # Calculate initial probability metrics
        train_metrics = self._calculate_probability_metrics(y_train_pred, y_train_true)
        val_metrics = self._calculate_probability_metrics(y_val_pred, y_val_true)
        
        # Calculate test metrics if provided
        test_metrics = None
        if y_test_pred is not None and y_test_true is not None:
            y_test_pred = np.array(y_test_pred)
            y_test_true = np.array(y_test_true)
            test_metrics = self._calculate_probability_metrics(y_test_pred, y_test_true)
        
        # Calculate ROC and PR curves for plotting
        train_curves = self._calculate_curves(y_train_pred, y_train_true)
        val_curves = self._calculate_curves(y_val_pred, y_val_true)
        test_curves = None
        if y_test_pred is not None and y_test_true is not None:
            test_curves = self._calculate_curves(y_test_pred, y_test_true)
        
        # Update AUC values from curves
        train_metrics.auc_roc = train_curves['auc_roc']
        train_metrics.auc_pr = train_curves['auc_pr']
        val_metrics.auc_roc = val_curves['auc_roc']
        val_metrics.auc_pr = val_curves['auc_pr']
        if test_metrics is not None and test_curves is not None:
            test_metrics.auc_roc = test_curves['auc_roc']
            test_metrics.auc_pr = test_curves['auc_pr']
            
        # Determine threshold
        threshold = self._determine_threshold(
            val_curves['roc_thresholds'], 
            y_val_pred, 
            y_val_true, 
            custom_threshold
        )
        
        # Apply threshold to calculate classification metrics
        self._update_classification_metrics(train_metrics, y_train_pred, y_train_true, threshold)
        self._update_classification_metrics(val_metrics, y_val_pred, y_val_true, threshold)
        if test_metrics is not None:
            self._update_classification_metrics(test_metrics, y_test_pred, y_test_true, threshold)
        
        # Create evaluation results object
        results = EvaluationResults(
            train=train_metrics,
            validation=val_metrics,
            threshold=threshold,
            test=test_metrics
        )
        
        # Plot curves if requested
        if self.plot_curves:
            self._plot_curves(train_curves, val_curves, test_curves, results)
        
        return results
    
    def _calculate_probability_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> DatasetMetrics:
        """Calculate probability-based metrics"""
        return DatasetMetrics(
            log_loss=log_loss(y_true, y_pred),
            brier_score=brier_score_loss(y_true, y_pred),
            auc_roc=0.0,  # Will be updated later
            auc_pr=0.0    # Will be updated later
        )
    
    def _calculate_curves(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate ROC and PR curves"""
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
        auc_roc = auc(fpr, tpr)
        
        # PR curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
        auc_pr = average_precision_score(y_true, y_pred)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds,
            'precision': precision,
            'recall': recall,
            'pr_thresholds': pr_thresholds,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr
        }
    
    def _determine_threshold(self, 
                            thresholds: np.ndarray, 
                            y_val_pred: np.ndarray, 
                            y_val_true: np.ndarray,
                            custom_threshold: Optional[float]) -> float:
        """Determine the optimal threshold to use"""
        if custom_threshold is not None:
            return custom_threshold
        
        if not self.find_optimal_threshold:
            return 0.5
        
        # Calculate F1 for each threshold on validation data
        f1_scores = []
        for threshold in thresholds:
            y_val_pred_binary = (y_val_pred >= threshold).astype(int)
            f1 = f1_score(y_val_true, y_val_pred_binary)
            f1_scores.append(f1)
        
        # Find threshold that maximizes F1
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx]
    
    def _update_classification_metrics(self, 
                                     metrics: DatasetMetrics, 
                                     y_pred: np.ndarray, 
                                     y_true: np.ndarray,
                                     threshold: float) -> None:
        """Update metrics with threshold-dependent classification metrics"""
        # Apply threshold
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate basic metrics
        metrics.accuracy = accuracy_score(y_true, y_pred_binary)
        metrics.precision = precision_score(y_true, y_pred_binary)
        metrics.recall = recall_score(y_true, y_pred_binary)
        metrics.f1_score = f1_score(y_true, y_pred_binary)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        metrics.confusion = ConfusionMetrics(
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn)
        )
        
        # Calculate additional metrics
        metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics.sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        metrics.balanced_accuracy = (metrics.sensitivity + metrics.specificity) / 2
        metrics.positive_predictive_value = metrics.precision  # Same as precision
        metrics.negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    def _plot_curves(self, 
                train_curves: Dict[str, np.ndarray],
                val_curves: Dict[str, np.ndarray],
                test_curves: Optional[Dict[str, np.ndarray]],
                results: EvaluationResults) -> None:
        """Plot ROC and PR curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # No need to update metrics here since they were already updated in evaluate()
        
        # ROC Curve
        ax1.plot(train_curves['fpr'], train_curves['tpr'], 
                label=f'Train (AUC = {results.train.auc_roc:.3f})')
        ax1.plot(val_curves['fpr'], val_curves['tpr'], 
                label=f'Validation (AUC = {results.validation.auc_roc:.3f})')
        
        # Add test ROC curve if test data is provided
        if test_curves is not None and results.test is not None:
            ax1.plot(test_curves['fpr'], test_curves['tpr'], 
                    label=f'Test (AUC = {results.test.auc_roc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend(loc="lower right")
        
        # PR Curve
        ax2.plot(train_curves['recall'], train_curves['precision'], 
                label=f'Train (AP = {results.train.auc_pr:.3f})')
        ax2.plot(val_curves['recall'], val_curves['precision'], 
                label=f'Validation (AP = {results.validation.auc_pr:.3f})')
        
        # Add test PR curve if test data is provided
        if test_curves is not None and results.test is not None:
            ax2.plot(test_curves['recall'], test_curves['precision'], 
                    label=f'Test (AP = {results.test.auc_pr:.3f})')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        
        # Add optimal threshold marker
        threshold = results.threshold
        
        # Find points on curves corresponding to optimal threshold
        train_opt_idx = np.argmin(np.abs(train_curves['roc_thresholds'] - threshold))
        val_opt_idx = np.argmin(np.abs(val_curves['roc_thresholds'] - threshold))
        
        # Mark on ROC curve
        ax1.scatter(train_curves['fpr'][train_opt_idx], train_curves['tpr'][train_opt_idx], 
                color='blue', marker='o', s=80, alpha=0.5)
        ax1.scatter(val_curves['fpr'][val_opt_idx], val_curves['tpr'][val_opt_idx], 
                color='orange', marker='o', s=80, alpha=0.5)
        
        # Add threshold annotation
        ax1.annotate(f'Threshold: {threshold:.3f}', 
                xy=(0.05, 0.05), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    y_train_true = np.random.randint(0, 2, 100)
    y_train_pred = np.random.random(100)
    y_val_true = np.random.randint(0, 2, 50)
    y_val_pred = np.random.random(50)
    y_test_true = np.random.randint(0, 2, 30)
    y_test_pred = np.random.random(30)
    
    # Example 1: Find optimal threshold and evaluate on all datasets
    print("EXAMPLE 1: Using optimal threshold from validation data")
    evaluator = Evaluator(find_optimal_threshold=True, plot_curves=True)
    results = evaluator.evaluate(
        y_train_pred, y_train_true, 
        y_val_pred, y_val_true, 
        y_test_pred, y_test_true
    )
    
    print(f"Optimal threshold: {results.threshold:.3f}")
    
    # Print metrics for each dataset
    for dataset_name, metrics in [
        ('Training', results.train), 
        ('Validation', results.validation), 
        ('Test', results.test)
    ]:
        if metrics is not None:
            print(f"\n{dataset_name} metrics:")
            for field_name, field_value in metrics.__dict__.items():
                if field_name != 'confusion' and isinstance(field_value, (int, float)):
                    print(f"  {field_name}: {field_value:.3f}")
    
    print("\nConfusion matrix (test):")
    for field_name, field_value in results.test.confusion.__dict__.items():
        print(f"  {field_name}: {field_value}")
    
    # Example 2: Use a custom threshold
    print("\n\nEXAMPLE 2: Using custom threshold")
    custom_thresh = 0.7
    evaluator = Evaluator(plot_curves=False)
    results = evaluator.evaluate(
        y_train_pred, y_train_true, 
        y_val_pred, y_val_true, 
        y_test_pred, y_test_true,
        custom_threshold=custom_thresh
    )
    
    print(f"Custom threshold: {results.threshold:.3f}")
    print("\nTest metrics with custom threshold:")
    for field_name, field_value in results.test.__dict__.items():
        if field_name != 'confusion' and isinstance(field_value, (int, float)):
            print(f"  {field_name}: {field_value:.3f}")