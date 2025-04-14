# experiments/experiment6_training_size.py
"""
Refactored Experiment 6: Effect of Training Data Size on Performance.

Goal: Evaluate how model performance changes as the percentage of available
training data increases, while keeping the test set constant.
1. For each training percentage (e.g., 10%, 20%, ..., 100%):
2.   For each fold:
3.     Prepare the full training and test sets for the fold.
4.     Subsample the *training* set according to the current percentage.
5.     Train the model(s) on the subsampled training data.
6.     Evaluate the model(s) on the *full* test set.
7. Plot performance metrics vs. training data percentage.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

from experiments import config
from experiments.data_preparation import prepare_experiment_data # Use this to get full fold data
# We won't use the standard runner directly, as we need to modify the training data inside the loop
from src.models import CatBoostModel, XGBoostModel, RandomForestModel, SVMModel, KNNModel # Import models directly
from src.evaluation.metrics import calculate_metrics
from src.utils.io import save_pickle, load_pickle
from src.utils.seeding import set_seed

# --- Configuration ---
# Define training percentages to test
TRAINING_PERCENTAGES = np.arange(0.1, 1.1, 0.1) # 10% to 100% in steps of 10%
MODELS_TO_TEST = config.MODEL_LIST # Or just ['catboost'] if replicating original exactly
N_FOLDS = config.N_FOLDS
RANDOM_SEED = config.RANDOM_SEED
set_seed(RANDOM_SEED) # Set seed for reproducibility of subsampling

# Data configuration: Use 'real' data as the base for training/testing?
# Or 'synthetic'? Let's assume 'real' based on the original script's use of PCA_fullDataExamples
# which likely operated on the base dataset without synthetic attacks initially. Adjust if needed.
BASE_DATA_TYPE = 'real'
# Set oversampling for the *full* training set before subsampling? Or apply after? Let's do before.
BASE_OVERSAMPLE = None # 'adasyn' or None

BASE_TRAIN_CONFIG = {'type': BASE_DATA_TYPE, 'oversample': BASE_OVERSAMPLE}
BASE_TEST_CONFIG = {'type': BASE_DATA_TYPE}


RESULTS_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment6_training_size_results.pkl')
PLOT_FILENAME_PATTERN = os.path.join(config.RESULTS_DIR, 'experiment6_training_size_{metric}.png')

# Map model names to classes (copied from runner.py for direct use)
MODEL_CLASSES = {
    'catboost': CatBoostModel,
    'xgboost': XGBoostModel,
    'rf': RandomForestModel,
    'svm': SVMModel,
    'knn': KNNModel,
}

# --- Main Execution ---
if __name__ == "__main__":
    print("\nStarting Refactored Experiment 6: Effect of Training Size")

    # --- Results Storage ---
    # results[percentage][model_name][metric_name] = [fold1_score, ...]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # --- Loop over Folds ---
    for fold_num in range(1, N_FOLDS + 1):
        print(f"\n===== Processing Fold {fold_num}/{N_FOLDS} =====")

        try:
            # 1. Prepare FULL training and test sets for this fold ONCE
            print(f"  Preparing full data for Fold {fold_num}...")
            (X_train_full, y_train_full), (X_test_full, y_test_full) = prepare_experiment_data(
                fold_id=fold_num,
                train_config=BASE_TRAIN_CONFIG,
                test_config=BASE_TEST_CONFIG
            )
            print(f"  Full Train: {X_train_full.shape}, Full Test: {X_test_full.shape}")

            if X_train_full.shape[0] < 2 or X_test_full.shape[0] == 0:
                 print(f"  Skipping Fold {fold_num} due to insufficient base data.")
                 continue

        except Exception as e:
            print(f"  Error preparing base data for Fold {fold_num}: {e}. Skipping fold.")
            # Add NaNs to results for all percentages/models for this fold if desired
            continue

        # --- Loop over Training Percentages ---
        for percentage in TRAINING_PERCENTAGES:
            perc_label = int(percentage * 100)
            print(f"\n  --- Processing Training Percentage: {perc_label}% ---")

            # 2. Subsample Training Data
            n_samples_to_take = int(np.ceil(X_train_full.shape[0] * percentage)) # Use ceil to ensure at least 1 sample
            if n_samples_to_take < 2: # Need at least 2 samples for train_test_split or some models
                 print(f"    Skipping {perc_label}%: Too few samples ({n_samples_to_take}).")
                 # Add NaNs to results for this percentage/models/fold
                 for model_name in MODELS_TO_TEST:
                      for metric in config.DEFAULT_METRICS:
                           results[perc_label][model_name][metric].append(np.nan)
                 continue

            print(f"    Subsampling {n_samples_to_take} training examples (Stratified)...")
            try:
                # Use train_test_split for stratified subsampling (trick: split off the portion we *don't* want)
                # If only one class in y_train_full, stratify will fail.
                unique_classes, class_counts = np.unique(y_train_full, return_counts=True)
                if len(unique_classes) < 2:
                    print(f"    Warning: Only one class ({unique_classes[0]}) present in full training data for fold {fold_num}. Using simple random sampling.")
                    indices = np.random.choice(X_train_full.shape[0], n_samples_to_take, replace=False)
                    X_train_sub = X_train_full[indices]
                    y_train_sub = y_train_full[indices]
                else:
                     # Check if n_samples_to_take is less than the number of classes (can cause issues)
                     if n_samples_to_take < len(unique_classes):
                          print(f"    Warning: Number of samples ({n_samples_to_take}) is less than number of classes ({len(unique_classes)}). Using simple random sampling.")
                          indices = np.random.choice(X_train_full.shape[0], n_samples_to_take, replace=False)
                          X_train_sub = X_train_full[indices]
                          y_train_sub = y_train_full[indices]
                     else:
                          # Calculate the size of the part to discard
                          discard_size = X_train_full.shape[0] - n_samples_to_take
                          if discard_size > 0:
                              # Split off the discard pile, keeping the rest
                              X_train_sub, _, y_train_sub, _ = train_test_split(
                                  X_train_full, y_train_full,
                                  test_size=discard_size, # Size to discard
                                  random_state=RANDOM_SEED,
                                  stratify=y_train_full
                              )
                          else: # percentage is 100%
                              X_train_sub, y_train_sub = X_train_full, y_train_full

            except ValueError as e:
                 print(f"    Error during stratified subsampling (likely due to small class size for {perc_label}%): {e}. Using simple random sampling.")
                 indices = np.random.choice(X_train_full.shape[0], n_samples_to_take, replace=False)
                 X_train_sub = X_train_full[indices]
                 y_train_sub = y_train_full[indices]

            print(f"    Subsampled Train Shape: {X_train_sub.shape}")

            # --- Loop over Models ---
            for model_name in MODELS_TO_TEST:
                print(f"      Model: {model_name}")
                try:
                    # 3. Load Hyperparameters (assuming based on BASE_DATA_TYPE)
                    hyperparams = config.load_hyperparameters(model_name, BASE_DATA_TYPE)

                    # 4. Initialize Model
                    model_class = MODEL_CLASSES.get(model_name)
                    if not model_class: raise ValueError(f"Unknown model: {model_name}")
                    model = model_class(params=hyperparams)

                    # 5. Train Model on Subsampled Data
                    model.fit(X_train_sub, y_train_sub)

                    # 6. Evaluate Model on Full Test Data
                    y_pred = model.predict(X_test_full)
                    y_proba = None
                    try:
                        if hasattr(model.model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test_full)[:, 1]
                        elif hasattr(model.model, 'decision_function') and model_name == 'svm':
                            y_proba = model.decision_function(X_test_full)
                    except AttributeError: pass # Ignore if proba/decision func not available

                    trial_metrics = calculate_metrics(y_test_full, y_pred, y_proba)
                    print(f"        Metrics: {trial_metrics}")

                    # 7. Store Results
                    for metric, value in trial_metrics.items():
                        results[perc_label][model_name][metric].append(value)

                except Exception as e:
                    print(f"      Error processing model {model_name} for {perc_label}%: {e}")
                    # Store NaNs for this model/percentage/fold
                    for metric in config.DEFAULT_METRICS:
                        results[perc_label][model_name][metric].append(np.nan)


    # --- End of Loops ---

    # --- Save Full Results ---
    print(f"\nSaving training size results to {RESULTS_FILENAME}...")
    # Convert defaultdicts to dicts for saving
    final_results = {
        perc: {model: dict(metrics) for model, metrics in models.items()}
        for perc, models in results.items()
    }
    save_pickle(final_results, RESULTS_FILENAME)

    # --- Plot Results ---
    print("\nGenerating training size plots...")
    percentages_sorted = sorted(final_results.keys())
    x_values = np.array(percentages_sorted)

    for metric in config.DEFAULT_METRICS:
        if metric == 'ROC-AUC': # Skip AUC if it wasn't calculated reliably
             # Check if any valid AUC values exist
             has_auc = any(np.isfinite(val) for perc_res in final_results.values()
                           for model_res in perc_res.values()
                           for val in model_res.get(metric, []))
             if not has_auc:
                  print(f"  Skipping plot for {metric}: No valid values found.")
                  continue

        print(f"  Plotting {metric} vs Training Size...")
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            for model_name in MODELS_TO_TEST:
                means = []
                stds = []
                for perc in percentages_sorted:
                    values = final_results.get(perc, {}).get(model_name, {}).get(metric, [])
                    if values: # Check if list is not empty after potential NaNs
                        means.append(np.nanmean(values))
                        stds.append(np.nanstd(values))
                    else:
                        means.append(np.nan) # Append NaN if no data for this point
                        stds.append(np.nan)

                # Plot only if there's valid data for this model
                if not np.all(np.isnan(means)):
                    means = np.array(means)
                    stds = np.array(stds)
                    ax.plot(x_values, means, label=model_name, marker='o', linestyle='-')
                    ax.fill_between(x_values, means - stds, means + stds, alpha=0.2)

            ax.set_title(f'{metric} vs. Training Data Percentage')
            ax.set_xlabel('Percentage of Training Data Used (%)')
            ax.set_ylabel(f'Average {metric} (across {N_FOLDS} folds)')
            ax.set_xticks(x_values) # Ensure ticks match percentages
            ax.legend()
            ax.grid(alpha=0.3)
            # ax.set_ylim(bottom=0.0) # Adjust y-axis limits if needed

            plot_filepath = PLOT_FILENAME_PATTERN.format(metric=metric.lower().replace('-', '_'))
            plt.tight_layout()
            plt.savefig(plot_filepath)
            print(f"  Plot saved to {plot_filepath}")
            # plt.close(fig) # Close figure

        except Exception as e:
            print(f"  Error generating plot for metric {metric}: {e}")


    print("\nExperiment 6 finished.")
    # plt.show() # Show plots if running interactively
