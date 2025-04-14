# experiments/experiment7_monthly_eval.py
"""
Refactored Experiment 7: Monthly Prediction and Aggregation Evaluation.

Goal: Evaluate model performance using a month-by-month prediction strategy.
1. Train a model (e.g., on monthly examples from a training fold).
2. For each consumer in the test set:
3.   Predict an anomaly score (probability) for each month using the trained model.
4. Aggregate these monthly scores per consumer (e.g., calculate mean confidence).
5. Classify consumers based on the aggregated score and/or the number of months
   exceeding a threshold (requires finding optimal thresholds).
6. Evaluate the final consumer-level classifications.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

from experiments import config
# Use data_preparation to get full fold data, then process monthly
from experiments.data_preparation import prepare_experiment_data
from src.models import CatBoostModel # Or other models
from src.evaluation.metrics import calculate_metrics # Use standard metrics for final eval
from src.utils.io import save_pickle, load_pickle
from src.utils.seeding import set_seed
# Need the specific preprocessing used by predict_each_month (pad_and_stats)
from src.data.preprocessing import add_statistical_features, pad_array

# --- Configuration ---
MODEL_NAME = 'catboost'
N_FOLDS = config.N_FOLDS
RANDOM_SEED = config.RANDOM_SEED
set_seed(RANDOM_SEED)

# Data configuration: Use 'real' data as base?
BASE_DATA_TYPE = 'real'
BASE_OVERSAMPLE = None # Or 'adasyn'
BASE_TRAIN_CONFIG = {'type': BASE_DATA_TYPE, 'oversample': BASE_OVERSAMPLE}
BASE_TEST_CONFIG = {'type': BASE_DATA_TYPE}

# Thresholds for grid search (based on original script)
CONFIDENCE_THRESHOLDS = np.arange(0.0, 1.05, 0.05) # 0 to 1.0 in steps of 0.05
MONTH_THRESHOLDS = np.arange(0, 35, 1) # 0 to 34 months

RESULTS_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment7_monthly_eval_results.pkl')
BEST_THRESHOLDS_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment7_best_thresholds.pkl')

# --- Helper Functions ---

def _predict_monthly_scores(model, X_full: np.ndarray, y_full: np.ndarray, consumer_ids: List[str], n_months: int) -> pd.DataFrame:
    """
    Predicts monthly anomaly scores for each consumer.

    Assumes X_full contains monthly examples stacked vertically, ordered by month, then consumer.
    y_full contains corresponding labels (needed to map back to consumers if order isn't guaranteed).
    consumer_ids provides the unique consumer IDs present in this dataset split.
    n_months is the expected number of months per consumer in this dataset.

    Returns:
        pd.DataFrame: Index=ConsumerID, Columns=MonthNumber (0 to n_months-1), Values=Anomaly Score
    """
    print("    Predicting monthly scores...")
    # Check if the number of examples matches expected consumers * months
    if X_full.shape[0] != len(consumer_ids) * n_months:
         print(f"Warning: Number of examples ({X_full.shape[0]}) doesn't match consumers*months ({len(consumer_ids)}*{n_months}). Score mapping might be inaccurate.")
         # Fallback or error handling needed here. Can we reliably reconstruct?
         # If examples are grouped by month first, then consumer, we can reshape.
         # Let's assume the order from create_monthly_examples: all examples for month 1, then month 2, ...
         # And within each month, consumers are in the order of the original DataFrame columns.
         # This is fragile. A better approach would be to pass the original DataFrame and predict month by month.
         # Let's try the DataFrame approach from the original script.

    # --- Alternative: Use DataFrame directly (closer to original `predict_each_month`) ---
    # This requires having access to the DataFrame corresponding to X_full/y_full
    # Let's assume we modify `prepare_experiment_data` or have another way to get it.
    # For now, simulate prediction on the numpy array assuming order.

    # Predict probabilities for all examples at once
    y_proba_all = None
    try:
        if hasattr(model.model, 'predict_proba'):
            y_proba_all = model.predict_proba(X_full)[:, 1] # Probability of positive class
        elif hasattr(model.model, 'decision_function') and model_name == 'svm':
             y_proba_all = model.decision_function(X_full) # Use decision values
    except AttributeError:
        print(f"    Error: Model {model_name} lacks predict_proba/decision_function needed for scores.")
        return pd.DataFrame() # Return empty DataFrame

    if y_proba_all is None:
        print("    Error: Could not get prediction probabilities/scores.")
        return pd.DataFrame()

    # Reshape scores based on assumed order (Fragile!)
    # Assumes create_monthly_examples stacks like:
    # [consumer1_month1, consumer2_month1, ..., consumerN_month1, consumer1_month2, ...] NO!
    # Original code stacks like:
    # [all_consumers_month1, all_consumers_month2, ...]
    # Let's assume X_full is [n_examples, n_features], y_full is [n_examples]
    # We need to know which example corresponds to which consumer and which month.
    # This information is lost when just passing X/y arrays.

    # ---> REVISED APPROACH: Pass the DataFrame corresponding to the test set <---
    # We need to adapt the flow to have access to the test DataFrame.
    # For now, we cannot implement this function correctly without that access.
    # Let's return an empty DataFrame and note this limitation.
    print("    Error: Cannot reliably map predicted scores back to consumers/months from numpy arrays alone.")
    print("    Refactoring needed to predict month-by-month on the DataFrame.")
    return pd.DataFrame() # Placeholder

def _find_best_thresholds(monthly_scores_df: pd.DataFrame, true_labels_series: pd.Series) -> Tuple[float, int]:
    """
    Performs a grid search to find the best confidence and month count thresholds
    based on maximizing F1 score on the *training* or *validation* data.

    Args:
        monthly_scores_df (pd.DataFrame): DataFrame of monthly scores (Consumers x Months).
        true_labels_series (pd.Series): True labels indexed by ConsumerID.

    Returns:
        Tuple[float, int]: best_confidence_threshold, best_month_threshold
    """
    print("    Finding best thresholds via grid search...")
    best_f1 = -1.0
    best_conf_thr = 0.5
    best_month_thr = 0

    # Align true labels with consumers in the scores dataframe
    consumers = monthly_scores_df.index
    y_true = true_labels_series.reindex(consumers).fillna(-1).astype(int).values # Fill missing labels?
    if -1 in y_true:
        print("    Warning: Some consumers in monthly scores missing from true labels.")

    for conf_thr in tqdm(CONFIDENCE_THRESHOLDS, desc="  Conf Threshold", leave=False):
        # Count anomalous months per consumer based on confidence threshold
        anomalous_months_count = (monthly_scores_df > conf_thr).sum(axis=1)

        for month_thr in MONTH_THRESHOLDS:
            # Predict final label based on month count threshold
            y_pred = (anomalous_months_count >= month_thr).astype(int)

            # Calculate F1 score for this threshold combination
            current_f1 = f1_score(y_true, y_pred, zero_division=0)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_conf_thr = conf_thr
                best_month_thr = month_thr

    print(f"    Best Thresholds Found: Confidence={best_conf_thr:.2f}, Months={best_month_thr} (F1={best_f1:.4f})")
    return best_conf_thr, best_month_thr


def _evaluate_with_thresholds(monthly_scores_df: pd.DataFrame, true_labels_series: pd.Series, conf_thr: float, month_thr: int) -> Dict[str, float]:
    """
    Evaluates consumer-level predictions using the chosen thresholds.
    """
    consumers = monthly_scores_df.index
    y_true = true_labels_series.reindex(consumers).fillna(-1).astype(int).values

    anomalous_months_count = (monthly_scores_df > conf_thr).sum(axis=1)
    y_pred = (anomalous_months_count >= month_thr).astype(int)

    # Use standard calculate_metrics (consumer-level evaluation)
    # We don't have consumer-level probability scores here easily, so AUC might be inaccurate/unavailable
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba=None) # Pass None for proba
    # Calculate consumer-level AUC based on average monthly score?
    avg_scores = monthly_scores_df.mean(axis=1).values
    try:
        consumer_auc = roc_auc_score(y_true, avg_scores)
        metrics['Consumer ROC-AUC (Avg Score)'] = consumer_auc
    except Exception as e:
        print(f"    Warning: Could not calculate consumer AUC based on avg score: {e}")
        metrics['Consumer ROC-AUC (Avg Score)'] = np.nan

    return metrics


# --- Main Execution ---
if __name__ == "__main__":
    print("\nStarting Refactored Experiment 7: Monthly Evaluation Strategy")

    # --- Results Storage ---
    # Store metrics per fold using the best thresholds found on validation set
    results_per_fold = []
    best_thresholds_per_fold = []

    # --- Loop over Folds ---
    for fold_num in range(1, N_FOLDS + 1):
        print(f"\n===== Processing Fold {fold_num}/{N_FOLDS} =====")
        fold_results = {}
        best_thresholds = None

        try:
            # 1. Prepare FULL training and test sets for this fold
            print(f"  Preparing full data for Fold {fold_num}...")
            # IMPORTANT: We need access to the original DataFrames to perform
            # month-by-month prediction correctly. `prepare_experiment_data`
            # currently returns numpy arrays. We need to modify it or add a
            # function to return the DataFrames and labels Series.
            # Let's assume we have a hypothetical function `get_fold_dataframes` for now.

            # --- Placeholder for getting DataFrames ---
            print("  Placeholder: Assuming get_fold_dataframes() returns train_df, test_df, train_labels, test_labels")
            # (train_df, train_labels), (test_df, test_labels) = get_fold_dataframes(fold_num, BASE_TRAIN_CONFIG, BASE_TEST_CONFIG)
            # For now, we cannot proceed without this data structure.
            print(f"  !!! Critical: Need access to test DataFrame for monthly prediction. Skipping Fold {fold_num}. !!!")
            results_per_fold.append({}) # Append empty results
            best_thresholds_per_fold.append(None)
            continue
            # --- End Placeholder ---

            # Split training data further for validation set to find thresholds
            # X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(...)
            # train_df_fold, val_df_fold, train_labels_fold, val_labels_fold = ... # Split DFs too

            # 2. Train Model (on X_train_fold, y_train_fold examples)
            print(f"  Training {MODEL_NAME}...")
            # (X_train_examples, y_train_examples), _ = prepare_experiment_data(...) # Get examples if needed
            # hyperparams = config.load_hyperparameters(MODEL_NAME, BASE_DATA_TYPE)
            # model = MODEL_CLASSES[MODEL_NAME](params=hyperparams)
            # model.fit(X_train_examples, y_train_examples) # Train on examples
            # --- Placeholder for Model Training ---
            print("  Placeholder: Model training would happen here.")
            model = None # Placeholder
            # --- End Placeholder ---


            # 3. Predict Monthly Scores on Validation Set (Requires DataFrame)
            # print("  Predicting monthly scores on Validation Set...")
            # val_monthly_scores = _predict_monthly_scores(model, val_df_fold) # Needs implementation

            # 4. Find Best Thresholds on Validation Set
            # print("  Finding best thresholds...")
            # best_conf_thr, best_month_thr = _find_best_thresholds(val_monthly_scores, val_labels_fold)
            # best_thresholds = (best_conf_thr, best_month_thr)
            # best_thresholds_per_fold.append(best_thresholds)
            # --- Placeholder for Threshold Finding ---
            print("  Placeholder: Threshold finding on validation set needed.")
            best_thresholds = (0.5, 1) # Default placeholder
            best_thresholds_per_fold.append(best_thresholds)
            # --- End Placeholder ---


            # 5. Predict Monthly Scores on Test Set (Requires DataFrame)
            # print("  Predicting monthly scores on Test Set...")
            # test_monthly_scores = _predict_monthly_scores(model, test_df) # Needs implementation
            # --- Placeholder for Test Prediction ---
            print("  Placeholder: Monthly prediction on test set needed.")
            # Create dummy scores for structure testing
            test_labels = pd.Series(np.random.randint(0,2,10), index=[f'C{i}' for i in range(10)]) # Dummy labels
            test_monthly_scores = pd.DataFrame(np.random.rand(10, 34), index=test_labels.index) # Dummy scores
            # --- End Placeholder ---


            # 6. Evaluate on Test Set using Best Thresholds
            print("  Evaluating on Test Set with best thresholds...")
            fold_results = _evaluate_with_thresholds(test_monthly_scores, test_labels, best_thresholds[0], best_thresholds[1])
            print(f"  Fold Test Metrics: {fold_results}")


        except Exception as e:
            print(f"  Error processing Fold {fold_num}: {e}")
            # Store empty results or NaNs for this fold
            fold_results = {metric: np.nan for metric in config.DEFAULT_METRICS}
            best_thresholds_per_fold.append(None)

        results_per_fold.append(fold_results)

    # --- Aggregate and Save Results ---
    print("\nAggregating results across folds...")
    aggregated_metrics = defaultdict(list)
    for fold_res in results_per_fold:
        if fold_res: # Check if fold processing succeeded
            for metric, value in fold_res.items():
                aggregated_metrics[metric].append(value)

    final_summary = {metric: {'mean': np.nanmean(values), 'std': np.nanstd(values)}
                     for metric, values in aggregated_metrics.items()}

    print("\nFinal Aggregated Metrics (Monthly Eval Strategy):")
    for metric, stats in final_summary.items():
        print(f"  {metric}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")

    print(f"\nSaving aggregated results to {RESULTS_FILENAME}...")
    save_pickle(final_summary, RESULTS_FILENAME)
    print(f"Saving best thresholds per fold to {BEST_THRESHOLDS_FILENAME}...")
    save_pickle(best_thresholds_per_fold, BEST_THRESHOLDS_FILENAME)

    # --- Plotting (Optional) ---
    # Could plot distribution of best thresholds, or final performance bars
    print("\nExperiment 7 finished.")
