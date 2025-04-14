# experiments/experiment7_monthly_eval.py
"""
Refactored Experiment 7: Monthly Prediction and Aggregation Evaluation.

Goal: Evaluate model performance using a month-by-month prediction strategy.
1. For each fold:
2.   Split the fold's training data into train/validation subsets.
3.   Prepare monthly examples for train/validation sets.
4.   Train a model on the training subset examples.
5.   Predict monthly scores for the validation DataFrame.
6.   Find optimal confidence/month thresholds using validation scores and labels.
7.   Predict monthly scores for the test DataFrame (from the fold split).
8.   Evaluate consumer-level predictions on the test set using the optimal thresholds.
9. Aggregate results across folds.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split # For validation split
from sklearn.metrics import f1_score, roc_auc_score

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

from experiments import config
# Use data_preparation ONLY to get the raw examples for training/validation
from experiments.data_preparation import prepare_experiment_data, _load_base_data_once
from src.data.fold_generator import get_fold # To get consumer IDs per fold
from src.data.preprocessing import create_monthly_examples # To create examples for training
from src.models import CatBoostModel # Or other models
from src.evaluation.metrics import calculate_metrics # Use standard metrics for final eval
from src.utils.io import save_pickle, load_pickle
from src.utils.seeding import set_seed

# --- Configuration ---
MODEL_NAME = 'catboost'
N_FOLDS = config.N_FOLDS
RANDOM_SEED = config.RANDOM_SEED
set_seed(RANDOM_SEED) # Set seed early

# Data configuration
BASE_DATA_TYPE = 'real' # Train/Val/Test based on real data structure
BASE_OVERSAMPLE = None # Or 'adasyn' applied to train subset examples
VALIDATION_SPLIT_SIZE = 0.2 # Use 20% of fold's training data for validation

# Thresholds for grid search
CONFIDENCE_THRESHOLDS = np.arange(0.0, 1.05, 0.05)
MONTH_THRESHOLDS = np.arange(0, 35, 1) # Assuming max ~34 months in data? Adjust if needed.

RESULTS_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment7_monthly_eval_results.pkl')
BEST_THRESHOLDS_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment7_best_thresholds.pkl')

# --- Helper Functions ---

def _predict_monthly_scores(model, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts monthly anomaly scores for each consumer in the target DataFrame.

    Args:
        model: Trained model instance with predict_proba/decision_function.
        target_df (pd.DataFrame): DataFrame (Time index, Consumer columns) to predict on.

    Returns:
        pd.DataFrame: Index=ConsumerID, Columns=MonthPeriod, Values=Anomaly Score
                      Returns empty DataFrame on error.
    """
    print(f"    Predicting monthly scores for {target_df.shape[1]} consumers...")
    if target_df.empty:
        return pd.DataFrame()

    consumer_ids = target_df.columns
    # Group by month using PeriodIndex for robust handling of month ends
    target_df_copy = target_df.copy() # Avoid modifying original
    target_df_copy['Year-Month'] = target_df_copy.index.to_period('M')
    unique_months = sorted(target_df_copy['Year-Month'].unique())

    # Initialize DataFrame to store scores
    monthly_scores = pd.DataFrame(index=consumer_ids, columns=unique_months, dtype=float)

    for month in tqdm(unique_months, desc="    Monthly Prediction", leave=False):
        month_data_df = target_df_copy[target_df_copy['Year-Month'] == month][consumer_ids] # Get data for this month only

        if month_data_df.empty:
            continue

        # Prepare examples for this month (consumers x days)
        # Need labels just as placeholders for create_monthly_examples structure
        dummy_labels = pd.Series(0, index=month_data_df.columns)
        X_month, _ = create_monthly_examples(month_data_df, dummy_labels, add_stats=True, pad_length=31) # Match training preprocessing

        if X_month.size == 0:
            print(f"      Warning: No examples generated for month {month}. Skipping.")
            continue

        # Predict probabilities/scores
        y_proba_month = None
        try:
            if hasattr(model.model, 'predict_proba'):
                y_proba_month = model.predict_proba(X_month)[:, 1] # Prob positive class
            elif hasattr(model.model, 'decision_function') and MODEL_NAME == 'svm':
                 y_proba_month = model.decision_function(X_month)
            else:
                 print(f"      Error: Model {MODEL_NAME} lacks predict_proba/decision_function.")
                 # Assign NaN or skip month? Assign NaN for now.
                 monthly_scores.loc[month_data_df.columns, month] = np.nan
                 continue

            # Store scores for the correct consumers for this month
            monthly_scores.loc[month_data_df.columns, month] = y_proba_month

        except Exception as e:
            print(f"      Error predicting for month {month}: {e}")
            monthly_scores.loc[month_data_df.columns, month] = np.nan # Assign NaN on error

    return monthly_scores


def _find_best_thresholds(monthly_scores_df: pd.DataFrame, true_labels_series: pd.Series) -> Tuple[float, int]:
    """ Finds best thresholds based on F1 score (Grid Search). """
    print("    Finding best thresholds via grid search on validation data...")
    best_f1 = -1.0
    best_conf_thr = 0.5
    best_month_thr = 0

    consumers = monthly_scores_df.index
    # Align labels, handle missing consumers if any
    y_true = true_labels_series.reindex(consumers).fillna(-1).astype(int)
    valid_indices = y_true != -1 # Only evaluate on consumers with known labels
    y_true = y_true[valid_indices].values
    monthly_scores_valid = monthly_scores_df.loc[valid_indices]

    if len(y_true) == 0:
         print("    Error: No valid consumers with labels found for threshold tuning.")
         return 0.5, 0 # Return default thresholds

    for conf_thr in tqdm(CONFIDENCE_THRESHOLDS, desc="    Conf Threshold", leave=False):
        anomalous_months_count = (monthly_scores_valid > conf_thr).sum(axis=1)
        for month_thr in MONTH_THRESHOLDS:
            y_pred = (anomalous_months_count >= month_thr).astype(int)
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_conf_thr = conf_thr
                best_month_thr = month_thr

    print(f"    Best Validation Thresholds: Confidence={best_conf_thr:.2f}, Months={best_month_thr} (F1={best_f1:.4f})")
    return best_conf_thr, best_month_thr


def _evaluate_with_thresholds(monthly_scores_df: pd.DataFrame, true_labels_series: pd.Series, conf_thr: float, month_thr: int) -> Dict[str, float]:
    """ Evaluates consumer-level predictions using chosen thresholds. """
    consumers = monthly_scores_df.index
    y_true = true_labels_series.reindex(consumers).fillna(-1).astype(int)
    valid_indices = y_true != -1
    y_true = y_true[valid_indices].values
    monthly_scores_valid = monthly_scores_df.loc[valid_indices]

    if len(y_true) == 0:
         print("    Error: No valid consumers with labels found for final evaluation.")
         return {metric: np.nan for metric in config.DEFAULT_METRICS + ['Consumer ROC-AUC (Avg Score)']}


    anomalous_months_count = (monthly_scores_valid > conf_thr).sum(axis=1)
    y_pred = (anomalous_months_count >= month_thr).astype(int)

    # Use standard calculate_metrics (consumer-level evaluation)
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba=None) # No single probability score

    # Calculate consumer-level AUC based on average monthly score
    avg_scores = monthly_scores_valid.mean(axis=1).values
    try:
        # Check if more than one class is present in y_true for AUC calculation
        if len(np.unique(y_true)) > 1:
            consumer_auc = roc_auc_score(y_true, avg_scores)
            metrics['Consumer ROC-AUC (Avg Score)'] = consumer_auc
        else:
             print("    Warning: Only one class present in y_true. Cannot calculate Consumer ROC-AUC.")
             metrics['Consumer ROC-AUC (Avg Score)'] = np.nan
    except Exception as e:
        print(f"    Warning: Could not calculate consumer AUC based on avg score: {e}")
        metrics['Consumer ROC-AUC (Avg Score)'] = np.nan

    return metrics


# --- Main Execution ---
if __name__ == "__main__":
    print("\nStarting Refactored Experiment 7: Monthly Evaluation Strategy")

    # --- Load Base Data Once ---
    try:
        _load_base_data_once()
        base_df = _BASE_DATA_CACHE["original_df"]
        all_labels = _BASE_DATA_CACHE["labels_series"]
        real_theft_ids = _BASE_DATA_CACHE["real_theft_ids"]
        all_benign_ids = _BASE_DATA_CACHE["all_benign_ids"]
    except Exception as e:
        print(f"Failed to load base data: {e}. Exiting.")
        sys.exit(1)

    # --- Results Storage ---
    results_per_fold = []
    best_thresholds_per_fold = []

    # --- Loop over Folds ---
    for fold_num in range(1, N_FOLDS + 1):
        print(f"\n===== Processing Fold {fold_num}/{N_FOLDS} =====")
        fold_results = {}
        best_thresholds = (0.5, 1) # Default

        try:
            # 1. Get consumer IDs for this fold
            benign_train_val_ids, benign_test_ids = get_fold(fold_num, config.FOLDS_FILE)
            test_consumer_ids = benign_test_ids + real_theft_ids
            test_df = base_df[test_consumer_ids]
            test_labels = all_labels.loc[test_consumer_ids]

            # 2. Split benign train/val IDs for validation set
            if len(benign_train_val_ids) < 2:
                 print("  Warning: Not enough benign consumers to create validation set. Using full train set for training and default thresholds.")
                 benign_train_ids = benign_train_val_ids
                 benign_val_ids = []
                 perform_threshold_tuning = False
            else:
                 benign_train_ids, benign_val_ids = train_test_split(
                     benign_train_val_ids, test_size=VALIDATION_SPLIT_SIZE, random_state=RANDOM_SEED
                 )
                 perform_threshold_tuning = True

            # Define consumers for training and validation DataFrames
            train_consumer_ids = benign_train_ids + real_theft_ids # Train on benign subset + real theft
            val_consumer_ids = benign_val_ids + real_theft_ids     # Validate on benign subset + real theft

            train_df = base_df[train_consumer_ids]
            train_labels = all_labels.loc[train_consumer_ids]

            if perform_threshold_tuning and len(benign_val_ids) > 0:
                 val_df = base_df[val_consumer_ids]
                 val_labels = all_labels.loc[val_consumer_ids]
            else: # Handle case with no validation set
                 val_df = pd.DataFrame()
                 val_labels = pd.Series(dtype=int)


            # 3. Prepare Training Examples (using create_monthly_examples)
            print(f"  Preparing training examples ({train_df.shape[1]} consumers)...")
            X_train_examples, y_train_examples = create_monthly_examples(train_df, train_labels)
            print(f"  Training examples shape: {X_train_examples.shape}")

            if X_train_examples.shape[0] < 2:
                 print(f"  Skipping Fold {fold_num}: Not enough training examples generated.")
                 results_per_fold.append({})
                 best_thresholds_per_fold.append(None)
                 continue

            # Optional: Apply oversampling to training examples
            if BASE_OVERSAMPLE == 'adasyn':
                 print("  Applying ADASYN to training examples...")
                 # X_train_examples, y_train_examples = apply_adasyn(X_train_examples, y_train_examples, random_state=RANDOM_SEED)
                 print("  ADASYN application placeholder.") # Placeholder

            # 4. Train Model
            print(f"  Training {MODEL_NAME}...")
            hyperparams = config.load_hyperparameters(MODEL_NAME, BASE_DATA_TYPE)
            model = MODEL_CLASSES[MODEL_NAME](params=hyperparams)
            model.fit(X_train_examples, y_train_examples)

            # 5. Find Best Thresholds using Validation Set (if possible)
            if perform_threshold_tuning and not val_df.empty:
                val_monthly_scores = _predict_monthly_scores(model, val_df)
                if not val_monthly_scores.empty:
                    best_conf_thr, best_month_thr = _find_best_thresholds(val_monthly_scores, val_labels)
                    best_thresholds = (best_conf_thr, best_month_thr)
                else:
                     print("  Warning: Failed to get validation scores. Using default thresholds.")
                     best_thresholds = (0.5, 1) # Default
            else:
                 print("  Skipping threshold tuning (no validation set). Using default thresholds.")
                 best_thresholds = (0.5, 1) # Default

            best_thresholds_per_fold.append(best_thresholds)

            # 6. Predict Monthly Scores on Test Set
            print("  Predicting monthly scores on Test Set...")
            test_monthly_scores = _predict_monthly_scores(model, test_df)

            # 7. Evaluate on Test Set using Best Thresholds
            if not test_monthly_scores.empty:
                print("  Evaluating on Test Set with determined thresholds...")
                fold_results = _evaluate_with_thresholds(test_monthly_scores, test_labels, best_thresholds[0], best_thresholds[1])
                print(f"  Fold {fold_num} Test Metrics: {fold_results}")
            else:
                 print("  Error: Failed to get test scores. Cannot evaluate fold.")
                 fold_results = {metric: np.nan for metric in config.DEFAULT_METRICS + ['Consumer ROC-AUC (Avg Score)']}


        except Exception as e:
            print(f"  Error processing Fold {fold_num}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            fold_results = {metric: np.nan for metric in config.DEFAULT_METRICS + ['Consumer ROC-AUC (Avg Score)']}
            best_thresholds_per_fold.append(None) # Mark thresholds as failed for this fold

        results_per_fold.append(fold_results)

    # --- Aggregate and Save Results ---
    print("\nAggregating results across folds...")
    aggregated_metrics = defaultdict(list)
    valid_fold_count = 0
    for fold_res in results_per_fold:
        if fold_res and not all(np.isnan(list(fold_res.values()))): # Check if fold has valid results
            valid_fold_count += 1
            for metric, value in fold_res.items():
                aggregated_metrics[metric].append(value)

    if valid_fold_count > 0:
         final_summary = {metric: {'mean': np.nanmean(values), 'std': np.nanstd(values)}
                          for metric, values in aggregated_metrics.items()}

         print(f"\nFinal Aggregated Metrics (Monthly Eval Strategy across {valid_fold_count} valid folds):")
         for metric, stats in final_summary.items():
             print(f"  {metric}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")

         print(f"\nSaving aggregated results to {RESULTS_FILENAME}...")
         save_pickle(final_summary, RESULTS_FILENAME)
    else:
         print("\nNo valid fold results obtained. Cannot aggregate or save summary.")


    print(f"Saving best thresholds per fold to {BEST_THRESHOLDS_FILENAME}...")
    save_pickle(best_thresholds_per_fold, BEST_THRESHOLDS_FILENAME)

    print("\nExperiment 7 finished.")
