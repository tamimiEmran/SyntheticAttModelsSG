# experiments/experiment2_additive.py
"""
Refactored Experiment 2 (Additive): Iterative Attack Addition for Subset Selection.

Goal: Find a good subset of synthetic attacks for training by starting with the
single best-performing attack and iteratively adding the attack that provides
the largest performance increase on the real test set.
1. Evaluate each single attack type individually (trained on synthetic, tested on real).
2. Select the best single attack as the starting point.
3. Iteratively add the remaining attack that, when combined with the current set,
   results in the highest performance on the real test set.
4. Track performance and the added attack at each step.
"""

import os
import sys
import numpy as np
import pickle
from collections import defaultdict
import copy # To avoid modifying lists directly

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

from experiments import config
from experiments.runner import run_single_trial
from src.evaluation.visualization import plot_metric_trends_with_errorbars
from src.utils.io import save_pickle
from src.utils.seeding import set_seed

# --- Configuration ---
MODEL_NAME = 'catboost' # Model used for evaluation
N_FOLDS = config.N_FOLDS # Or 3? Let's use config for consistency
RANDOM_SEED = config.RANDOM_SEED
set_seed(RANDOM_SEED)

ALL_AVAILABLE_ATTACKS = config.ATTACK_IDS_ALL # All attacks to consider
METRIC_TO_OPTIMIZE = 'ROC-AUC' # Metric used to decide which attack to add
TEST_CONFIG = {'type': 'real'} # Always test on the real set

RESULTS_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment2_additive_results.pkl')
PLOT_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment2_additive_auc_trend.png')

# --- Main Execution ---
if __name__ == "__main__":
    print("\nStarting Refactored Experiment 2 (Additive): Iterative Attack Addition")

    # --- Initialization ---
    current_optimal_set = []
    remaining_attacks = copy.deepcopy(ALL_AVAILABLE_ATTACKS)
    addition_history = [] # Stores tuples: (added_attack, resulting_set, avg_metric, std_metric, raw_metrics_list)

    # --- Iterative Addition Loop ---
    while len(remaining_attacks) > 0:
        iteration_num = len(current_optimal_set) + 1
        print(f"\n===== Iteration {iteration_num}: Finding best attack to add to set {current_optimal_set} =====")

        best_avg_metric_this_iter = -np.inf # Initialize with worst possible score
        best_attack_to_add = None
        best_addition_raw_metrics = []
        best_addition_std = np.nan

        # --- Try adding each remaining attack ---
        for i, attack_to_try_adding in enumerate(remaining_attacks):
            print(f"  --- Testing addition of: {attack_to_try_adding} ({i+1}/{len(remaining_attacks)}) ---")
            candidate_attack_set = current_optimal_set + [attack_to_try_adding]

            # Evaluate performance with this candidate set
            current_addition_fold_metrics = []
            train_cfg_candidate = {'type': 'synthetic', 'attack_ids': candidate_attack_set, 'oversample': None}

            for fold_num in range(1, N_FOLDS + 1):
                metrics = run_single_trial(
                    fold_id=fold_num,
                    model_name=MODEL_NAME,
                    train_config=train_cfg_candidate,
                    test_config=TEST_CONFIG, # Fixed real test set
                    load_params_if_none=True
                )
                if metrics:
                     current_addition_fold_metrics.append(metrics.get(METRIC_TO_OPTIMIZE, np.nan))
                else:
                     current_addition_fold_metrics.append(np.nan) # Mark failed fold

            avg_metric_candidate = np.nanmean(current_addition_fold_metrics)
            std_metric_candidate = np.nanstd(current_addition_fold_metrics)
            print(f"    Performance with {candidate_attack_set}: Avg {METRIC_TO_OPTIMIZE}={avg_metric_candidate:.4f}")

            # Check if this addition is the best so far in this iteration
            if avg_metric_candidate > best_avg_metric_this_iter:
                best_avg_metric_this_iter = avg_metric_candidate
                best_attack_to_add = attack_to_try_adding
                best_addition_raw_metrics = current_addition_fold_metrics
                best_addition_std = std_metric_candidate

        # --- End of trying all additions for this iteration ---

        if best_attack_to_add is None:
            # This might happen if all additions fail or result in NaN scores
            if not remaining_attacks: # Should not happen if loop condition is correct
                 print("Error: No remaining attacks but loop continued.")
            else:
                 print(f"Warning: Could not determine best attack to add in iteration {iteration_num}. Choosing first remaining: {remaining_attacks[0]}")
                 # Fallback: just add the first remaining attack (or handle error differently)
                 best_attack_to_add = remaining_attacks[0]
                 # We need metrics for this fallback choice - re-run or store NaN? Let's store NaN for now.
                 best_avg_metric_this_iter = np.nan
                 best_addition_std = np.nan
                 best_addition_raw_metrics = [np.nan] * N_FOLDS # Placeholder

        # Add the best attack found to the optimal set and remove from remaining
        print(f"\n  Best attack to add in this iteration: {best_attack_to_add}")
        print(f"  Resulting Performance ({METRIC_TO_OPTIMIZE}): Avg={best_avg_metric_this_iter:.4f}, Std={best_addition_std:.4f}")
        current_optimal_set.append(best_attack_to_add)
        remaining_attacks.remove(best_attack_to_add)

        # Record history
        addition_history.append(
            (best_attack_to_add, copy.deepcopy(current_optimal_set), best_avg_metric_this_iter, best_addition_std, best_addition_raw_metrics)
        )

    print(f"\nFinal optimal set (additive order): {current_optimal_set}")

    # --- Save Results ---
    print(f"\nSaving additive history results to {RESULTS_FILENAME}...")
    try:
        results_to_save = {
            'initial_attacks': ALL_AVAILABLE_ATTACKS,
            'metric_optimized': METRIC_TO_OPTIMIZE,
            'addition_history': addition_history # List of (added, resulting_set, avg, std, raw_list)
        }
        save_pickle(results_to_save, RESULTS_FILENAME)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

    # --- Plot Results ---
    print("\nGenerating additive trend plot...")
    try:
        # Extract data for plotting
        plot_labels = [item[0] for item in addition_history] # Attack added
        plot_raw_data = [item[4] for item in addition_history] # List of lists of raw fold metrics
        plot_avg_data = [item[2] for item in addition_history] # Average metric per step

        # Create sensible x-axis labels
        plot_x_labels = [f"Added {label}\n(Set Size {i+1})" for i, label in enumerate(plot_labels)]

        fig, ax = plt.subplots(figsize=(14, 8)) # Wider plot for more labels
        plot_metric_trends_with_errorbars(
            data_list=plot_raw_data,
            x_labels=plot_x_labels,
            title=f'Iterative Attack Addition ({MODEL_NAME} on Real Test Set)',
            xlabel='Step (Attack Added)',
            ylabel=f'Average {METRIC_TO_OPTIMIZE}',
            ax=ax
        )
        # Customize plot further if needed
        ax.tick_params(axis='x', rotation=70)
        plt.tight_layout()
        plt.savefig(PLOT_FILENAME)
        print(f"Additive trend plot saved to {PLOT_FILENAME}")
        # plt.show()

    except Exception as e:
        print(f"Error generating plot: {e}")

    print("\nExperiment 2 (Additive) finished.")
