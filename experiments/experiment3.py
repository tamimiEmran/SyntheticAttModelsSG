# experiments/experiment3.py
"""
Refactored Experiment 3: Comparison of Different Models on Real vs. Synthetic Data.

Goal: Compare the performance of various ML models when trained and tested on:
1. Real Data: Train on real theft + benign, Test on real theft + benign.
2. Synthetic Data: Train on synthetic theft (all attacks) + benign, Test on synthetic theft + benign.
3. Synthetic-Trained on Real Test: Train on synthetic theft + benign, Test on real theft + benign.
"""

import os
import sys
import numpy as np
import pickle
from collections import defaultdict
import copy

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

from experiments import config
from experiments.runner import run_single_trial
from src.evaluation.visualization import plot_comparison_bar, plot_metric_bars # Using the new plotting functions

# --- Configuration ---
MODELS_TO_TEST = config.MODEL_LIST # ['catboost', 'xgboost', 'rf', 'svm', 'knn']
N_FOLDS = config.N_FOLDS
SYNTHETIC_ATTACK_IDS = config.ATTACK_IDS_ALL # Use all attacks for synthetic training/testing

RESULTS_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment3_model_comparison_results.pkl')
PLOT_AUC_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment3_auc_comparison.png')
PLOT_METRICS_REAL_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment3_metrics_real.png')
PLOT_METRICS_SYNTH_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment3_metrics_synth.png')
PLOT_METRICS_SYNTH_ON_REAL_FILENAME = os.path.join(config.RESULTS_DIR, 'experiment3_metrics_synth_on_real.png')


# --- Main Execution ---
if __name__ == "__main__":
    print("\nStarting Refactored Experiment 3: Model Comparison (Real vs. Synthetic)")

    # --- Results Storage ---
    # results[scenario][model_name][metric_name] = [fold1_score, ...]
    # Scenarios: 'Real', 'Synthetic', 'SynthOnReal'
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # --- Loop over Folds ---
    for fold_num in range(1, N_FOLDS + 1):
        print(f"\n===== Processing Fold {fold_num}/{N_FOLDS} =====")

        # --- Define Data Configurations for this fold ---
        # Config for training on REAL data
        train_cfg_real = {'type': 'real', 'oversample': None} # Or 'adasyn'/'smote'
        # Config for testing on REAL data
        test_cfg_real = {'type': 'real'}

        # Config for training on SYNTHETIC data (using all attacks)
        train_cfg_synth = {'type': 'synthetic', 'attack_ids': SYNTHETIC_ATTACK_IDS, 'oversample': None} # Or 'adasyn'/'smote'
        # Config for testing on SYNTHETIC data (using all attacks)
        test_cfg_synth = {'type': 'synthetic', 'attack_ids': SYNTHETIC_ATTACK_IDS}

        # --- Loop over Models ---
        for model_name in MODELS_TO_TEST:
            print(f"\n  --- Processing Model: {model_name} ---")

            # --- Scenario 1: Train on Real, Test on Real ---
            print("    Scenario: Train Real, Test Real")
            metrics_real = run_single_trial(
                fold_id=fold_num,
                model_name=model_name,
                train_config=train_cfg_real,
                test_config=test_cfg_real,
                load_params_if_none=True # Load params based on 'real' data type
            )
            if metrics_real:
                for metric, value in metrics_real.items():
                    results['Real'][model_name][metric].append(value)
            else: # Handle trial failure
                 for metric in config.DEFAULT_METRICS:
                      results['Real'][model_name][metric].append(np.nan)


            # --- Scenario 2: Train on Synthetic, Test on Synthetic ---
            print("    Scenario: Train Synthetic, Test Synthetic")
            metrics_synth = run_single_trial(
                fold_id=fold_num,
                model_name=model_name,
                train_config=train_cfg_synth,
                test_config=test_cfg_synth,
                load_params_if_none=True # Load params based on 'synthetic' data type
            )
            if metrics_synth:
                for metric, value in metrics_synth.items():
                    results['Synthetic'][model_name][metric].append(value)
            else:
                 for metric in config.DEFAULT_METRICS:
                      results['Synthetic'][model_name][metric].append(np.nan)


            # --- Scenario 3: Train on Synthetic, Test on Real ---
            print("    Scenario: Train Synthetic, Test Real")
            # We need to run this again because test_config is different
            # Ideally, runner could take multiple test sets after one training.
            metrics_synth_on_real = run_single_trial(
                fold_id=fold_num,
                model_name=model_name,
                train_config=train_cfg_synth, # Train synthetic
                test_config=test_cfg_real,    # Test real
                load_params_if_none=True # Load params based on 'synthetic' data type
            )
            if metrics_synth_on_real:
                for metric, value in metrics_synth_on_real.items():
                    results['SynthOnReal'][model_name][metric].append(value)
            else:
                 for metric in config.DEFAULT_METRICS:
                      results['SynthOnReal'][model_name][metric].append(np.nan)

    # --- End of Loops ---

    # --- Save Results ---
    print(f"\nSaving experiment results to {RESULTS_FILENAME}...")
    # Convert defaultdicts to dicts for saving
    final_results = {
        scenario: {model: dict(metrics) for model, metrics in models.items()}
        for scenario, models in results.items()
    }
    try:
        with open(RESULTS_FILENAME, 'wb') as f:
            pickle.dump(final_results, f)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results: {e}")

    # --- Plot Results ---
    print("\nGenerating plots...")

    # 1. AUC Comparison Plot (Real vs Synthetic vs SynthOnReal)
    try:
        # Prepare data for the comparison plot (requires specific structure)
        plot_data_auc = defaultdict(lambda: defaultdict(list))
        for model in MODELS_TO_TEST:
            plot_data_auc[model]['Real'] = final_results.get('Real', {}).get(model, {}).get('ROC-AUC', [])
            plot_data_auc[model]['Synthetic'] = final_results.get('Synthetic', {}).get(model, {}).get('ROC-AUC', [])
            plot_data_auc[model]['SynthOnReal'] = final_results.get('SynthOnReal', {}).get(model, {}).get('ROC-AUC', [])

        # Need a custom plotting function or adapt plot_comparison_bar for multiple bars per group
        fig_auc, ax_auc = plt.subplots(figsize=(15, 8))
        n_models = len(MODELS_TO_TEST)
        x = np.arange(n_models)
        width = 0.25 # Width for each bar

        means_real = [np.nanmean(plot_data_auc[m]['Real']) for m in MODELS_TO_TEST]
        stds_real = [np.nanstd(plot_data_auc[m]['Real']) for m in MODELS_TO_TEST]
        means_synth = [np.nanmean(plot_data_auc[m]['Synthetic']) for m in MODELS_TO_TEST]
        stds_synth = [np.nanstd(plot_data_auc[m]['Synthetic']) for m in MODELS_TO_TEST]
        means_sor = [np.nanmean(plot_data_auc[m]['SynthOnReal']) for m in MODELS_TO_TEST]
        stds_sor = [np.nanstd(plot_data_auc[m]['SynthOnReal']) for m in MODELS_TO_TEST]

        rects1 = ax_auc.bar(x - width, means_real, width, yerr=stds_real, label='Train Real, Test Real', capsize=5)
        rects2 = ax_auc.bar(x, means_synth, width, yerr=stds_synth, label='Train Synth, Test Synth', capsize=5)
        rects3 = ax_auc.bar(x + width, means_sor, width, yerr=stds_sor, label='Train Synth, Test Real', capsize=5)

        ax_auc.set_ylabel('Average ROC-AUC')
        ax_auc.set_xlabel('Model')
        ax_auc.set_title('Model ROC-AUC Comparison: Real vs. Synthetic Training/Testing')
        ax_auc.set_xticks(x)
        ax_auc.set_xticklabels(MODELS_TO_TEST)
        ax_auc.legend()
        ax_auc.grid(axis='y', alpha=0.3)
        ax_auc.set_ylim(bottom=0.4) # Adjust y-axis start if needed
        fig_auc.tight_layout()
        plt.savefig(PLOT_AUC_FILENAME)
        print(f"AUC comparison plot saved to {PLOT_AUC_FILENAME}")
        # plt.close(fig_auc) # Close figure

    except Exception as e:
        print(f"Error generating AUC comparison plot: {e}")

    # 2. Grouped Metric Bars for each scenario
    metrics_to_plot = ['Precision', 'Recall', 'F1']
    try:
        if 'Real' in final_results:
            fig_real, ax_real = plt.subplots(figsize=(12, 7))
            plot_metric_bars(
                results_dict=final_results['Real'],
                metrics_to_plot=metrics_to_plot,
                title='Model Performance (Train Real, Test Real)',
                ax=ax_real
            )
            plt.savefig(PLOT_METRICS_REAL_FILENAME)
            print(f"Real metrics plot saved to {PLOT_METRICS_REAL_FILENAME}")
            # plt.close(fig_real)
    except Exception as e:
        print(f"Error generating Real metrics plot: {e}")

    try:
        if 'Synthetic' in final_results:
            fig_synth, ax_synth = plt.subplots(figsize=(12, 7))
            plot_metric_bars(
                results_dict=final_results['Synthetic'],
                metrics_to_plot=metrics_to_plot,
                title='Model Performance (Train Synthetic, Test Synthetic)',
                ax=ax_synth
            )
            plt.savefig(PLOT_METRICS_SYNTH_FILENAME)
            print(f"Synthetic metrics plot saved to {PLOT_METRICS_SYNTH_FILENAME}")
            # plt.close(fig_synth)
    except Exception as e:
        print(f"Error generating Synthetic metrics plot: {e}")

    try:
        if 'SynthOnReal' in final_results:
            fig_sor, ax_sor = plt.subplots(figsize=(12, 7))
            plot_metric_bars(
                results_dict=final_results['SynthOnReal'],
                metrics_to_plot=metrics_to_plot,
                title='Model Performance (Train Synthetic, Test Real)',
                ax=ax_sor
            )
            plt.savefig(PLOT_METRICS_SYNTH_ON_REAL_FILENAME)
            print(f"SynthOnReal metrics plot saved to {PLOT_METRICS_SYNTH_ON_REAL_FILENAME}")
            # plt.close(fig_sor)
    except Exception as e:
        print(f"Error generating SynthOnReal metrics plot: {e}")

    # Optional: Add Precision-Recall curve plots if needed

    print("\nExperiment 3 finished.")
    # plt.show() # Show all plots at the end if desired
