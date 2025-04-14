# experiments/experiment2.py
"""
Refactored Experiment 2: Iterative Attack Removal for Optimal Subset Selection.

Goal: Find the subset of synthetic attacks for training that maximizes
performance on the real test set using backward elimination.
1. Start with all synthetic attacks.
2. Evaluate performance on the real test set (average across folds).
3. Iteratively remove the single attack whose removal results in the highest
   performance on the real test set.
4. Track performance and the removed attack at each step.
"""

import os
import sys
import numpy as np
import pickle
from collections import defaultdict
import copy # To avoid modifying lists directly during iteration

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

from experiments import config
from experiments.runner import run_single_trial
from src.evaluation.visualization import plot_metric_trends_with_errorbars

# --- Configuration ---
MODEL_NAME = 'catboost' # Model used for evaluation
N_FOLDS = config.N_FOLDS
INITIAL_ATTACKS = config.ATTACK_IDS_ALL # Start with all defined attacks
METRIC_TO_OPTIMIZE = 'ROC-AUC' # Metric used to decide which attack to remove
RESULTS_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment2_removal_results.pkl')
PLOT_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment2_removal_auc_trend.png')

# --- Main Execution ---
if __name__ == "__main__":
    print("\nStarting Refactored Experiment 2: Iterative Attack Removal")

    # --- Initialization ---
    current_attack_set = copy.deepcopy(INITIAL_ATTACKS)
    removal_history = [] # Stores tuples: (removed_attack, resulting_attack_set, avg_metric, std_metric, raw_metrics_list)

    # --- Evaluate Initial State (All Attacks) ---
    print("\n===== Evaluating Initial State (All Attacks) =====")
    initial_fold_metrics = []
    train_cfg_initial = {'type': 'synthetic', 'attack_ids': current_attack_set, 'oversample': None}
    test_cfg_initial = {'type': 'real'}

    for fold_num in range(1, N_FOLDS + 1):
        metrics = run_single_trial(
            fold_id=fold_num,
            model_name=MODEL_NAME,
            train_config=train_cfg_initial,
            test_config=test_cfg_initial,
            load_params_if_none=True
        )
        if metrics: # Check if trial succeeded
             initial_fold_metrics.append(metrics.get(METRIC_TO_OPTIMIZE, np.nan))
        else:
             initial_fold_metrics.append(np.nan) # Mark failed fold

    initial_avg_metric = np.nanmean(initial_fold_metrics)
    initial_std_metric = np.nanstd(initial_fold_metrics)
    removal_history.append(
        ("None (Initial)", copy.deepcopy(current_attack_set), initial_avg_metric, initial_std_metric, initial_fold_metrics)
    )
    print(f"Initial Performance ({METRIC_TO_OPTIMIZE}): Avg={initial_avg_metric:.4f}, Std={initial_std_metric:.4f}")

    # --- Iterative Removal Loop ---
    while len(current_attack_set) > 1:
        print(f"\n===== Iteration: Removing from {len(current_attack_set)} attacks =====")
        best_avg_metric_this_iter = -np.inf # Initialize with worst possible score
        best_attack_to_remove = None
        best_removal_raw_metrics = []
        best_removal_std = np.nan

        # --- Try removing each attack currently in the set ---
        for i, attack_to_try_removing in enumerate(current_attack_set):
            print(f"  --- Testing removal of: {attack_to_try_removing} ({i+1}/{len(current_attack_set)}) ---")
            candidate_attack_set = [att for att in current_attack_set if att != attack_to_try_removing]

            if not candidate_attack_set: # Should not happen if len > 1, but good check
                print("    Cannot remove last attack, skipping.")
                continue

            # Evaluate performance with this candidate set
            current_removal_fold_metrics = []
            train_cfg_candidate = {'type': 'synthetic', 'attack_ids': candidate_attack_set, 'oversample': None}
            test_cfg_candidate = {'type': 'real'}

            for fold_num in range(1, N_FOLDS + 1):
                metrics = run_single_trial(
                    fold_id=fold_num,
                    model_name=MODEL_NAME,
                    train_config=train_cfg_candidate,
                    test_config=test_cfg_candidate,
                    load_params_if_none=True
                )
                if metrics:
                     current_removal_fold_metrics.append(metrics.get(METRIC_TO_OPTIMIZE, np.nan))
                else:
                     current_removal_fold_metrics.append(np.nan)

            avg_metric_candidate = np.nanmean(current_removal_fold_metrics)
            std_metric_candidate = np.nanstd(current_removal_fold_metrics)
            print(f"    Performance after removing {attack_to_try_removing}: Avg {METRIC_TO_OPTIMIZE}={avg_metric_candidate:.4f}")

            # Check if this removal is the best so far in this iteration
            if avg_metric_candidate > best_avg_metric_this_iter:
                best_avg_metric_this_iter = avg_metric_candidate
                best_attack_to_remove = attack_to_try_removing
                best_removal_raw_metrics = current_removal_fold_metrics
                best_removal_std = std_metric_candidate

        # --- End of trying all removals for this iteration ---

        if best_attack_to_remove is None:
            print("Error: Could not determine best attack to remove. Stopping.")
            break # Exit loop if something went wrong

        # Permanently remove the best attack found
        print(f"\n  Best attack to remove in this iteration: {best_attack_to_remove}")
        print(f"  Resulting Performance ({METRIC_TO_OPTIMIZE}): Avg={best_avg_metric_this_iter:.4f}, Std={best_removal_std:.4f}")
        current_attack_set.remove(best_attack_to_remove)

        # Record history
        removal_history.append(
            (best_attack_to_remove, copy.deepcopy(current_attack_set), best_avg_metric_this_iter, best_removal_std, best_removal_raw_metrics)
        )

    print(f"\nFinal attack set: {current_attack_set}")

    # --- Save Results ---
    print(f"\nSaving removal history results to {RESULTS_FILENAME}...")
    try:
        results_to_save = {
            'initial_attacks': INITIAL_ATTACKS,
            'metric_optimized': METRIC_TO_OPTIMIZE,
            'removal_history': removal_history # List of (removed, remaining_set, avg, std, raw_list)
        }
        with open(RESULTS_FILENAME, 'wb') as f:
            pickle.dump(results_to_save, f)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

    # --- Plot Results ---
    print("\nGenerating removal trend plot...")
    try:
        # Extract data for plotting
        plot_labels = [item[0] for item in removal_history] # Attack removed (or "None")
        plot_raw_data = [item[4] for item in removal_history] # List of lists of raw fold metrics

        # Adjust labels for clarity
        plot_x_labels = ["All Attacks"] + [f"Removed {label}" for label in plot_labels[1:]]

        fig, ax = plt.subplots(figsize=(12, 7))
        plot_metric_trends_with_errorbars(
            data_list=plot_raw_data,
            x_labels=plot_x_labels,
            title=f'Iterative Attack Removal ({MODEL_NAME} on Real Test Set)',
            xlabel='Step (Attack Removed)',
            ylabel=f'Average {METRIC_TO_OPTIMIZE}',
            ax=ax
        )
        # Customize plot further if needed
        ax.tick_params(axis='x', rotation=70)
        plt.tight_layout()
        plt.savefig(PLOT_FILENAME)
        print(f"Removal trend plot saved to {PLOT_FILENAME}")
        # plt.show()

    except Exception as e:
        print(f"Error generating plot: {e}")

    print("\nExperiment 2 finished.")
