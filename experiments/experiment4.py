# experiments/experiment4_transferability.py
"""
Refactored Experiment 4: Attack Transferability Matrix.

Goal: Evaluate how well a model trained on a single synthetic attack type
generalizes to detecting other synthetic attack types.
1. For each 'train_attack_id':
2.   Train a model using only benign data + data synthetically altered with 'train_attack_id'.
3.   For each 'test_attack_id':
4.     Test the trained model on a dataset containing only benign data + data
         synthetically altered with 'test_attack_id'.
5. Generate a heatmap of the results (e.g., AUC).
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

from experiments import config
from experiments.runner import run_single_trial

# --- Configuration ---
MODEL_NAME = 'catboost' # Model used for evaluation
N_FOLDS = config.N_FOLDS # Use fewer folds (e.g., 3) to speed up if needed, like original
# N_FOLDS = 3 # Use 3 folds like the original trainOnEachTestOnEach.py? Or stick to 10? Let's use config.N_FOLDS for consistency.

ATTACK_IDS_TO_ITERATE = config.ATTACK_IDS_ALL # Iterate through all attacks for train/test
METRIC_TO_PLOT = 'ROC-AUC'
RESULTS_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment4_transferability_results.pkl')
HEATMAP_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment4_transferability_heatmap.png')

# --- Main Execution ---
if __name__ == "__main__":
    print("\nStarting Refactored Experiment 4: Attack Transferability Matrix")

    # --- Results Storage ---
    # results[train_attack_id][test_attack_id][metric] = [fold1_score, ...]
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # --- Outer Loop: Training Attack ---
    for train_attack_id in ATTACK_IDS_TO_ITERATE:
        print(f"\n===== Training Attack: {train_attack_id} =====")

        # --- Inner Loop: Testing Attack ---
        for test_attack_id in ATTACK_IDS_TO_ITERATE:
            print(f"  --- Testing Attack: {test_attack_id} ---")

            # --- Loop over Folds ---
            # This is computationally expensive (Train*Test*Fold*Model)
            # Consider averaging results directly if fold-level detail isn't strictly needed for the final heatmap
            for fold_num in range(1, N_FOLDS + 1):
                print(f"    Fold {fold_num}/{N_FOLDS}")

                # Define configurations for this specific trial
                # Train on single synthetic attack
                train_cfg = {
                    'type': 'synthetic',
                    'attack_ids': [train_attack_id],
                    'oversample': None
                }
                # Test on single synthetic attack
                test_cfg = {
                    'type': 'synthetic',
                    'attack_ids': [test_attack_id]
                }

                # Run the trial
                trial_metrics = run_single_trial(
                    fold_id=fold_num,
                    model_name=MODEL_NAME,
                    train_config=train_cfg,
                    test_config=test_cfg,
                    load_params_if_none=True # Load params based on 'synthetic' type
                )

                # Store results for this train/test pair and fold
                if trial_metrics:
                    for metric, value in trial_metrics.items():
                        results[train_attack_id][test_attack_id][metric].append(value)
                else: # Handle trial failure
                    for metric in config.DEFAULT_METRICS:
                         results[train_attack_id][test_attack_id][metric].append(np.nan)


            # Optional: Print average metric for this train/test pair after all folds
            avg_metric_val = np.nanmean(results[train_attack_id][test_attack_id].get(METRIC_TO_PLOT, []))
            print(f"    Avg {METRIC_TO_PLOT} (Train {train_attack_id} / Test {test_attack_id}): {avg_metric_val:.4f}")


    # --- End of Loops ---

    # --- Save Full Results ---
    print(f"\nSaving full transferability results to {RESULTS_FILENAME}...")
    # Convert defaultdicts to dicts for saving
    final_results = {
        train_att: {
            test_att: {metric: vals for metric, vals in metrics.items()}
            for test_att, metrics in test_attacks.items()
        }
        for train_att, test_attacks in results.items()
    }
    try:
        with open(RESULTS_FILENAME, 'wb') as f:
            pickle.dump(final_results, f)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

    # --- Process for Heatmap ---
    print("\nProcessing results for heatmap...")
    heatmap_data = pd.DataFrame(
        index=ATTACK_IDS_TO_ITERATE, # Rows: Trained on Attack
        columns=ATTACK_IDS_TO_ITERATE # Cols: Tested on Attack
    )

    for train_id in ATTACK_IDS_TO_ITERATE:
        for test_id in ATTACK_IDS_TO_ITERATE:
            metric_values = final_results.get(train_id, {}).get(test_id, {}).get(METRIC_TO_PLOT, [])
            if metric_values:
                heatmap_data.loc[train_id, test_id] = np.nanmean(metric_values)
            else:
                heatmap_data.loc[train_id, test_id] = np.nan

    heatmap_data = heatmap_data.astype(float) # Ensure numeric type for plotting

    # --- Plot Heatmap ---
    print("Generating heatmap...")
    try:
        plt.figure(figsize=(14, 11)) # Adjust size as needed
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f", # Format annotations
            cmap="RdYlGn", # Red-Yellow-Green colormap (good for performance)
            linewidths=.5,
            vmin=0.4, # Set min value for color scale (adjust as needed)
            vmax=1.0, # Set max value for color scale
            cbar_kws={'label': f'Average {METRIC_TO_PLOT}'} # Label for the color bar
        )
        plt.title(f'{MODEL_NAME}: Trained on Single Attack (Rows) vs. Tested on Single Attack (Cols)')
        plt.xlabel('Attack Type Tested On')
        plt.ylabel('Attack Type Trained On')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(HEATMAP_FILENAME)
        print(f"Heatmap saved to {HEATMAP_FILENAME}")
        # plt.show()

    except Exception as e:
        print(f"Error generating heatmap: {e}")

    print("\nExperiment 4 finished.")
