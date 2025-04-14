# experiments/experiment1.py (Conceptual Refactoring using new helpers)

import os
import sys
import numpy as np
import pickle
from collections import defaultdict

# --- Add project root to sys.path ---
# ... (same as before) ...

from experiments import config
from experiments.runner import run_single_trial
from src.evaluation.visualization import plot_comparison_bar # Or the direct plotting used before

# --- Configuration ---
MODEL_NAME = 'catboost' # Focus of this experiment
N_FOLDS = config.N_FOLDS
ATTACK_IDS_TO_PROCESS = config.ATTACK_IDS_ALL
RESULTS_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment1_refactored_results.pkl')
PLOT_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment1_refactored_auc_comparison.png')

# --- Main Execution ---
if __name__ == "__main__":
    print("\nStarting Refactored Experiment 1: Individual Attack Evaluation")

    # Store results: results[attack_id][metric_name] = [fold1_score, ...]
    results = defaultdict(lambda: defaultdict(list))

    # --- Loop over Attack Types (Outer loop specific to Exp1) ---
    for attack_id in ATTACK_IDS_TO_PROCESS:
        print(f"\n===== Processing Attack Type: {attack_id} =====")

        # --- Loop over Folds ---
        for fold_num in range(1, N_FOLDS + 1):

            # Define configurations for this specific trial
            train_cfg_synth = {
                'type': 'synthetic',
                'attack_ids': [attack_id], # Train only on this attack
                'oversample': None # Or 'adasyn' / 'smote' if desired
            }
            test_cfg_synth = {
                'type': 'synthetic',
                'attack_ids': [attack_id] # Test on the same synthetic attack
            }
            test_cfg_real = {
                'type': 'real' # Test on real theft data + benign test consumers
            }

            # Run trial: Train on Synthetic, Test on Synthetic
            metrics_synth = run_single_trial(
                fold_id=fold_num,
                model_name=MODEL_NAME,
                train_config=train_cfg_synth,
                test_config=test_cfg_synth,
                # hyperparams=None, # Let runner load them
                load_params_if_none=True
            )
            if metrics_synth: # Only append if trial succeeded
                results[attack_id]['synth_test_auc'].append(metrics_synth.get('ROC-AUC', np.nan))
                # Store other metrics if needed: results[attack_id]['synth_metrics'].append(metrics_synth)


            # Run trial: Train on Synthetic, Test on Real
            # NOTE: Re-running the training is inefficient here.
            # A better approach would be to train once and test on both.
            # The current `run_single_trial` doesn't support this easily.
            # Let's modify `run_single_trial` or add a new runner function later if needed.
            # For now, we'll re-train for simplicity of demonstrating the structure.
            metrics_real = run_single_trial(
                fold_id=fold_num,
                model_name=MODEL_NAME,
                train_config=train_cfg_synth, # Same training data
                test_config=test_cfg_real,   # Different test data
                # hyperparams=None,
                load_params_if_none=True
            )
            if metrics_real: # Only append if trial succeeded
                results[attack_id]['real_test_auc'].append(metrics_real.get('ROC-AUC', np.nan))
                # Store other metrics if needed: results[attack_id]['real_metrics'].append(metrics_real)


        print(f"  Avg Synth Test AUC for Attack {attack_id}: {np.nanmean(results[attack_id]['synth_test_auc']):.4f}")
        print(f"  Avg Real Test AUC for Attack {attack_id}: {np.nanmean(results[attack_id]['real_test_auc']):.4f}")

    # --- Save Results ---
    print(f"\nSaving experiment results to {RESULTS_FILENAME}...")
    # Convert defaultdict back to dict for saving if needed
    final_results = {k: dict(v) for k, v in results.items()}
    try:
        with open(RESULTS_FILENAME, 'wb') as f:
            pickle.dump(final_results, f)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

    # --- Plot Results ---
    # ... (Use the same plotting logic as before, reading from final_results) ...
    print("\nPlotting generation...")
    # ... (plotting code using final_results) ...
    print("\nExperiment 1 finished.")
