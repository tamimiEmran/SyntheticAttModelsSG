# experiments/experiment5_robustness.py
"""
Refactored Experiment 5: Robustness of All-Attacks Model.

Goal: Evaluate how well a model trained on ALL synthetic attack types performs
when tested against datasets generated using only ONE specific synthetic attack type at a time.
1. Train a single model using benign data + data synthetically altered with ALL attack types.
2. For each 'test_attack_id':
3.   Test the trained model on a dataset containing only benign data + data
       synthetically altered with that specific 'test_attack_id'.
4. Generate a bar plot showing the performance (e.g., AUC) for each test attack.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

from experiments import config
from experiments.runner import run_single_trial
from src.evaluation.visualization import plot_comparison_bar # Use the standard bar plot

# --- Configuration ---
MODEL_NAME = 'catboost' # Model used for evaluation
N_FOLDS = config.N_FOLDS # Or 3 if matching original compareAttacksAusgrid exactly
# N_FOLDS = 3

# Training configuration is fixed: train on all synthetic attacks
TRAIN_ATTACK_IDS = config.ATTACK_IDS_ALL
TRAIN_CONFIG = {
    'type': 'synthetic',
    'attack_ids': TRAIN_ATTACK_IDS,
    'oversample': None # Or 'adasyn'/'smote'
}

# We will iterate through these attack IDs for testing
TEST_ATTACK_IDS_INDIVIDUAL = config.ATTACK_IDS_ALL

METRIC_TO_PLOT = 'ROC-AUC'
RESULTS_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment5_robustness_results.pkl')
PLOT_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment5_robustness_barplot.png')

# --- Main Execution ---
if __name__ == "__main__":
    print("\nStarting Refactored Experiment 5: Robustness of All-Attacks Model")

    # --- Results Storage ---
    # results[test_attack_id][metric] = [fold1_score, ...]
    results = defaultdict(lambda: defaultdict(list))

    # --- Loop over Individual Test Attacks ---
    for test_attack_id in TEST_ATTACK_IDS_INDIVIDUAL:
        print(f"\n===== Testing Attack: {test_attack_id} =====")

        # Define the test configuration for this specific attack
        test_cfg = {
            'type': 'synthetic',
            'attack_ids': [test_attack_id] # Test on only this attack
        }

        # --- Loop over Folds ---
        for fold_num in range(1, N_FOLDS + 1):
            print(f"  Fold {fold_num}/{N_FOLDS}")

            # Run the trial: Train on ALL attacks, Test on SINGLE attack
            # The training part is the same for all test_attack_ids within a fold,
            # but the current runner trains each time. This is inefficient but simple.
            # Optimization: Train once per fold outside the test_attack_id loop,
            # then loop through test_attack_id calling only model.predict/evaluate.
            # For now, using the existing runner for simplicity:
            trial_metrics = run_single_trial(
                fold_id=fold_num,
                model_name=MODEL_NAME,
                train_config=TRAIN_CONFIG, # Fixed training config
                test_config=test_cfg,      # Variable testing config
                load_params_if_none=True   # Load params based on 'synthetic' type
            )

            # Store results for this test attack and fold
            if trial_metrics:
                for metric, value in trial_metrics.items():
                    results[test_attack_id][metric].append(value)
            else: # Handle trial failure
                for metric in config.DEFAULT_METRICS:
                     results[test_attack_id][metric].append(np.nan)

        # Optional: Print average metric for this test attack after all folds
        avg_metric_val = np.nanmean(results[test_attack_id].get(METRIC_TO_PLOT, []))
        print(f"  Avg {METRIC_TO_PLOT} (Tested on {test_attack_id}): {avg_metric_val:.4f}")

    # --- End of Loops ---

    # --- Save Full Results ---
    print(f"\nSaving robustness results to {RESULTS_FILENAME}...")
    # Convert defaultdicts to dicts for saving
    final_results = {
        test_att: {metric: vals for metric, vals in metrics.items()}
        for test_att, metrics in results.items()
    }
    try:
        with open(RESULTS_FILENAME, 'wb') as f:
            pickle.dump(final_results, f)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

    # --- Plot Results ---
    print("\nGenerating robustness bar plot...")
    try:
        # Prepare data for plot_comparison_bar: Dict[TestAttackID, Dict[Metric, List[float]]]
        plot_data = {
            test_att: {METRIC_TO_PLOT: metrics.get(METRIC_TO_PLOT, [])}
            for test_att, metrics in final_results.items()
        }

        fig, ax = plt.subplots(figsize=(15, 8))
        plot_comparison_bar(
            results_dict=plot_data,
            metric=METRIC_TO_PLOT,
            title=f'{MODEL_NAME} Trained on All Attacks vs. Tested on Individual Attacks',
            sort_by_metric=True, # Sort bars by performance
            ax=ax
        )
        ax.set_xlabel('Individual Attack Type Tested On')
        ax.set_ylabel(f'Average {METRIC_TO_PLOT}')
        plt.tight_layout()
        plt.savefig(PLOT_FILENAME)
        print(f"Robustness plot saved to {PLOT_FILENAME}")
        # plt.show()

    except Exception as e:
        print(f"Error generating plot: {e}")

    print("\nExperiment 5 finished.")
