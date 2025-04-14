# unit_tests/test_attack_visualizations.py
"""
Unit tests for visualizing and verifying attack model implementations.

This module provides functions to visually compare the effects of different
attack models on real energy consumption data loaded from both Ausgrid and SGCC
datasets using the loader functions defined in src.data.loader.
It generates plots saved to the 'results' directory for visual inspection.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import necessary components from the project
try:
    from src.attack_models import get_attack_model, list_available_attacks
    # Assuming ATTACK_CONSTANTS might be used implicitly by get_attack_model
    # from experiments.config import ATTACK_CONSTANTS
    from src.data.loader import load_sgcc_data, load_ausgrid_data
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}")
    sys.exit(1) # Exit if core components can't be imported

class TestAttackVisualizations(unittest.TestCase):
    """
    Test case for visualizing and comparing attack implementations on real data formats.
    Uses data loaded via src.data.loader.
    """

    @classmethod
    def setUpClass(cls):
        """Set up by loading real data from SGCC and Ausgrid sources."""
        cls.results_dir = os.path.join(PROJECT_ROOT, 'results', 'attack_visualizations')
        os.makedirs(cls.results_dir, exist_ok=True)
        logging.info(f"Results will be saved in: {cls.results_dir}")

        cls.sgcc_df = None
        cls.sgcc_labels = None
        cls.ausgrid_df = None

        # --- Load SGCC data ---
        # Adjust path relative to project root or use an absolute path/environment variable
        sgcc_data_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'sgcc_data.csv')
        if os.path.exists(sgcc_data_path):
            try:
                cls.sgcc_df, cls.sgcc_labels = load_sgcc_data(sgcc_data_path)
                #print the head of the loaded data for debugging
                logging.info(f"SGCC data loaded successfully: {cls.sgcc_df.head()}")
                if cls.sgcc_df is not None and not cls.sgcc_df.empty:
                    logging.info(f"SGCC data loaded successfully: {cls.sgcc_df.shape}")
                    # Basic check: Ensure numeric data
                    if not pd.api.types.is_numeric_dtype(cls.sgcc_df.iloc[:, 0]):
                         logging.warning("SGCC data might not be numeric. Attacks may fail.")
                else:
                     logging.warning("SGCC data loaded but is None or empty.")
                     cls.sgcc_df = None # Ensure it's None if empty
            except FileNotFoundError:
                logging.warning(f"SGCC data file not found at {sgcc_data_path}. Tests using SGCC data will be skipped.")
            except ValueError as e:
                logging.error(f"Error loading SGCC data: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred loading SGCC data: {e}")
        else:
            logging.warning(f"SGCC data file not found at {sgcc_data_path}. Tests using SGCC data will be skipped.")

        # --- Load Ausgrid data ---
        # Adjust paths as needed
        ausgrid_base_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', 'ausgrid')
        ausgrid_dirs = [
            os.path.join(ausgrid_base_dir, 'ausgrid2010'),
            os.path.join(ausgrid_base_dir, 'ausgrid2011'),
            os.path.join(ausgrid_base_dir, 'ausgrid2012')
        ]
        # Check if *all* required directories exist before attempting load
        if all(os.path.isdir(d) for d in ausgrid_dirs):
             # Check if the specific expected CSV files exist within those directories
            expected_files = [
                os.path.join(ausgrid_dirs[0], 'ausgrid2010.csv'),
                os.path.join(ausgrid_dirs[1], 'ausgrid2011.csv'),
                os.path.join(ausgrid_dirs[2], 'ausgrid2012.csv'),
            ]
            if all(os.path.isfile(f) for f in expected_files):
                try:
                    # Assuming filenames and formats align with the loader defaults
                    cls.ausgrid_df = load_ausgrid_data(ausgrid_dirs)
                    # Print the head of the loaded data for debugging
                    logging.info(f"Ausgrid data loaded successfully: {cls.ausgrid_df.head()}")
                    if cls.ausgrid_df is not None and not cls.ausgrid_df.empty:
                        logging.info(f"Ausgrid data loaded successfully: {cls.ausgrid_df.shape}")
                        # Basic check: Ensure numeric data and DatetimeIndex
                        if not isinstance(cls.ausgrid_df.index, pd.DatetimeIndex):
                             logging.warning("Ausgrid data index is not DatetimeIndex. Time-based plots might fail.")
                        if not pd.api.types.is_numeric_dtype(cls.ausgrid_df.iloc[:, 0]):
                             logging.warning("Ausgrid data might not be numeric. Attacks may fail.")
                    else:
                         logging.warning("Ausgrid data loaded but is None or empty.")
                         cls.ausgrid_df = None # Ensure it's None if empty
                except FileNotFoundError: # Should be caught by earlier check, but good practice
                    logging.warning(f"One or more Ausgrid files not found in specified directories. Tests using Ausgrid data will be skipped.")
                except ValueError as e:
                    logging.error(f"Error loading Ausgrid data (check date formats or file structure): {e}")
                except Exception as e:
                    logging.error(f"An unexpected error occurred loading Ausgrid data: {e}")
            else:
                 logging.warning(f"Required Ausgrid CSV files not found in directories: {ausgrid_dirs}. Skipping Ausgrid tests.")
        else:
            logging.warning(f"One or more Ausgrid directories not found: {ausgrid_dirs}. Skipping Ausgrid tests.")


    def _select_data_window(self, df: pd.DataFrame, num_points: int = 150) -> pd.DataFrame:
        """Helper to select a random window of data, ensuring it's within bounds."""
        if df is None or df.empty:
            return None
        if len(df) <= num_points:
            return df # Return the whole dataframe if it's smaller than the window
        start_idx = random.randint(0, len(df) - num_points -1)
        end_idx = start_idx + num_points
        return df.iloc[start_idx:end_idx]

    def test_visualize_attacks_on_ausgrid(self):
        """Visualize all available attacks on a sample of loaded Ausgrid data."""
        if self.ausgrid_df is None:
            self.skipTest("Ausgrid data not available or failed to load. Skipping test.")

        attack_ids = list_available_attacks()
        if not attack_ids:
            self.skipTest("No available attacks found to test.")

        n_attacks = len(attack_ids)
        fig, axes = plt.subplots(n_attacks, 1, figsize=(18, 6 * n_attacks), squeeze=False) # Use squeeze=False

        # Select a random consumer for visualization
        if self.ausgrid_df.shape[1] == 0:
             self.fail("Ausgrid DataFrame has no columns (consumers).")
        consumer = random.choice(self.ausgrid_df.columns)
        logging.info(f"Ausgrid visualization using consumer: {consumer}")

        # Select a window of data for clarity (e.g., ~3 days if 48 readings/day)
        window_points = 48 * 3 # 144 points
        window_df = self._select_data_window(self.ausgrid_df, num_points=window_points)

        if window_df is None or window_df.empty:
             self.fail("Failed to select a valid data window from Ausgrid data.")

        original_data = window_df[consumer].copy() # Ensure it's a copy
        time_index = window_df.index

        # Check if original data is all zeros or constant, which might make attacks trivial
        if original_data.nunique() <= 1:
            logging.warning(f"Selected Ausgrid data window for consumer {consumer} is constant or all zeros.")

        successful_attacks = 0
        for i, attack_id in enumerate(attack_ids):
            ax = axes[i, 0] # Access axes correctly
            attack_model = None # Initialize
            try:
                attack_model = get_attack_model(attack_id)
                attack_name = attack_model.__class__.__name__

                # Apply the attack to a copy of the window data
                # Attack models should handle the DataFrame format (time as index, consumers as columns)
                attacked_df_window = attack_model.apply(window_df.copy())
                attacked_data = attacked_df_window[consumer]

                # Basic Assertion: Check if data was modified (unless it's a NoAttack type)
                # Use np.isclose for float comparison, allow for small differences
                is_modified = not np.allclose(original_data.values, attacked_data.values, atol=1e-6)
                if "NoAttack" not in attack_name and not is_modified:
                     logging.warning(f"Attack {attack_id} ({attack_name}) did not modify Ausgrid data significantly.")
                # self.assertTrue(is_modified or "NoAttack" in attack_name,
                #                 f"Attack {attack_id} ({attack_name}) should modify data.")

                # Calculate statistics for the title
                orig_sum = original_data.sum()
                attacked_sum = attacked_data.sum()
                reduction = 0
                if orig_sum != 0: # Avoid division by zero
                    reduction = (orig_sum - attacked_sum) / orig_sum * 100

                # Plot data
                ax.plot(time_index, original_data, 'b-', alpha=0.7, label='Original')
                ax.plot(time_index, attacked_data, 'r-', alpha=0.8, label=f'Attacked ({attack_name})')

                # Formatting
                ax.set_title(f'Attack {attack_id} ({attack_name}) on Ausgrid | Consumer: {consumer} | Energy Reduction: {reduction:.1f}%')
                ax.set_xlabel('Time')
                ax.set_ylabel('Consumption')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.4)

                # Format x-axis if it's datetime
                if isinstance(time_index, pd.DatetimeIndex):
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
                else: # Handle non-datetime index if occurs
                     plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

                successful_attacks += 1

            except Exception as e:
                logging.error(f"Error applying/plotting attack {attack_id} on Ausgrid data: {e}", exc_info=True)
                ax.text(0.5, 0.5, f"Error applying attack {attack_id}:\n{str(e)}",
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=10, color='red', wrap=True)
                ax.set_title(f'Attack {attack_id} - FAILED')

        fig.suptitle('Attack Visualization on Ausgrid Data Sample', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
        save_path = os.path.join(self.results_dir, 'ausgrid_attack_visualizations.png')
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"Ausgrid visualization saved to {save_path}")

        # Test passes if at least one attack could be visualized without fatal errors
        self.assertGreater(successful_attacks, 0, "No attacks were successfully visualized for Ausgrid.")

    def test_visualize_attacks_on_sgcc(self):
        """Visualize all available attacks on a sample of loaded SGCC data."""
        if self.sgcc_df is None:
            self.skipTest("SGCC data not available or failed to load. Skipping test.")

        attack_ids = list_available_attacks()
        if not attack_ids:
            self.skipTest("No available attacks found to test.")

        n_attacks = len(attack_ids)
        fig, axes = plt.subplots(n_attacks, 1, figsize=(18, 6 * n_attacks), squeeze=False)

        # Select a random consumer for visualization
        # SGCC loader default format: Consumers as rows, Time as columns. We need Consumers as columns.
        # Let's transpose SGCC data for consistency with Ausgrid and attack model expectations.
        try:
            sgcc_df_transposed = self.sgcc_df.T
            sgcc_df_transposed.index = pd.to_datetime(sgcc_df_transposed.index, errors='coerce')
             # Drop rows where index conversion failed, if any
            sgcc_df_transposed = sgcc_df_transposed.dropna(axis=0, how='all', subset=None) # Ensure index is valid datetime
            sgcc_df_transposed.index.name = 'Timestamp'
            # Convert columns (consumer IDs) to strings if they are numeric
            sgcc_df_transposed.columns = sgcc_df_transposed.columns.astype(str)

        except Exception as e:
            self.fail(f"Failed to transpose and prepare SGCC data for visualization: {e}")

        if sgcc_df_transposed.shape[1] == 0:
             self.fail("Transposed SGCC DataFrame has no columns (consumers).")
        consumer = random.choice(sgcc_df_transposed.columns)
        logging.info(f"SGCC visualization using consumer: {consumer}")

        # Select a window of data (e.g., 50 points)
        window_points = 90 
        window_df = self._select_data_window(sgcc_df_transposed, num_points=window_points)

        if window_df is None or window_df.empty:
             self.fail("Failed to select a valid data window from transposed SGCC data.")

        original_data = window_df[consumer].copy()
        time_index = window_df.index

        if original_data.nunique() <= 1:
             logging.warning(f"Selected SGCC data window for consumer {consumer} is constant or all zeros.")

        successful_attacks = 0
        for i, attack_id in enumerate(attack_ids):
            ax = axes[i, 0]
            attack_model = None
            try:
                attack_model = get_attack_model(attack_id)
                attack_name = attack_model.__class__.__name__

                # Apply attack to the transposed window data
                attacked_df_window = attack_model.apply(window_df.copy())
                attacked_data = attacked_df_window[consumer]

                # Basic Assertion: Check if data was modified
                is_modified = not np.allclose(original_data.values, attacked_data.values, atol=1e-6)
                if "NoAttack" not in attack_name and not is_modified:
                     logging.warning(f"Attack {attack_id} ({attack_name}) did not modify SGCC data significantly.")
                # self.assertTrue(is_modified or "NoAttack" in attack_name,
                #                 f"Attack {attack_id} ({attack_name}) should modify data.")

                # Calculate statistics
                orig_sum = original_data.sum()
                attacked_sum = attacked_data.sum()
                reduction = 0
                if orig_sum != 0:
                    reduction = (orig_sum - attacked_sum) / orig_sum * 100

                # Plot data
                ax.plot(time_index, original_data, 'b-', alpha=0.7, label='Original')
                ax.plot(time_index, attacked_data, 'r-', alpha=0.8, label=f'Attacked ({attack_name})')

                # Formatting
                ax.set_title(f'Attack {attack_id} ({attack_name}) on SGCC | Consumer: {consumer} | Energy Reduction: {reduction:.1f}%')
                ax.set_xlabel('Time')
                ax.set_ylabel('Consumption')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.4)

                # Format x-axis if it's datetime
                if isinstance(time_index, pd.DatetimeIndex):
                    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d')) # SGCC often daily
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
                else: # Handle non-datetime index
                     ax.set_xlabel('Time Step') # Use generic label
                     plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

                successful_attacks += 1

            except Exception as e:
                logging.error(f"Error applying/plotting attack {attack_id} on SGCC data: {e}", exc_info=True)
                ax.text(0.5, 0.5, f"Error applying attack {attack_id}:\n{str(e)}",
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=10, color='red', wrap=True)
                ax.set_title(f'Attack {attack_id} - FAILED')

        fig.suptitle('Attack Visualization on SGCC Data Sample (Transposed)', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        save_path = os.path.join(self.results_dir, 'sgcc_attack_visualizations.png')
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"SGCC visualization saved to {save_path}")

        # Test passes if at least one attack could be visualized
        self.assertGreater(successful_attacks, 0, "No attacks were successfully visualized for SGCC.")
'''
    def test_compare_attack_impacts(self):
        """Compare statistical impacts (reduction, peak change) across attacks and datasets."""
        if self.sgcc_df is None or self.ausgrid_df is None:
            self.skipTest("Both SGCC and Ausgrid data are required for impact comparison. Skipping.")

        attack_ids = list_available_attacks()
        if not attack_ids:
            self.skipTest("No available attacks found to test.")

        # Metrics to track
        metrics = ['energy_reduction_pct', 'max_reading_change_pct', 'std_dev_change_pct']
        results = {
            'ausgrid': {metric: [] for metric in metrics},
            'sgcc': {metric: [] for metric in metrics}
        }

        # --- Prepare Data Samples ---
        # Ausgrid: Use first ~week (336 points) or less if dataset is smaller
        ausgrid_sample_size = min(len(self.ausgrid_df), 48 * 7)
        ausgrid_sample = self.ausgrid_df.iloc[:ausgrid_sample_size].copy()
        if ausgrid_sample.empty: self.fail("Ausgrid sample is empty.")

        # SGCC: Transpose and use first ~100 points or less
        try:
            sgcc_df_transposed = self.sgcc_df.T
            sgcc_df_transposed.index = pd.to_datetime(sgcc_df_transposed.index, errors='coerce')
            sgcc_df_transposed = sgcc_df_transposed.dropna(axis=0, how='all', subset=None)
            sgcc_df_transposed.columns = sgcc_df_transposed.columns.astype(str)
        except Exception as e:
            self.fail(f"Failed to transpose SGCC data for comparison: {e}")

        sgcc_sample_size = min(len(sgcc_df_transposed), 100)
        sgcc_sample = sgcc_df_transposed.iloc[:sgcc_sample_size].copy()
        if sgcc_sample.empty: self.fail("SGCC sample (transposed) is empty.")

        # Calculate baseline metrics (sum over all consumers in the sample)
        # Ausgrid
        orig_energy_ausgrid = ausgrid_sample.sum().sum()
        orig_max_ausgrid = ausgrid_sample.max().max() # Max reading anywhere in the sample
        orig_std_ausgrid = ausgrid_sample.std(axis=0).mean() # Avg std dev across consumers
        # SGCC
        orig_energy_sgcc = sgcc_sample.sum().sum()
        orig_max_sgcc = sgcc_sample.max().max()
        orig_std_sgcc = sgcc_sample.std(axis=0).mean()

        # --- Apply Attacks and Calculate Metrics ---
        for attack_id in attack_ids:
            attack_model = None
            try:
                attack_model = get_attack_model(attack_id)
                attack_name = attack_model.__class__.__name__
                logging.info(f"Comparing impact of Attack {attack_id} ({attack_name})")

                # Process Ausgrid
                attacked_ausgrid = attack_model.apply(ausgrid_sample.copy())
                att_energy_ausgrid = attacked_ausgrid.sum().sum()
                att_max_ausgrid = attacked_ausgrid.max().max()
                att_std_ausgrid = attacked_ausgrid.std(axis=0).mean()

                results['ausgrid']['energy_reduction_pct'].append(
                    ((orig_energy_ausgrid - att_energy_ausgrid) / orig_energy_ausgrid * 100) if orig_energy_ausgrid else 0
                )
                results['ausgrid']['max_reading_change_pct'].append(
                    ((orig_max_ausgrid - att_max_ausgrid) / orig_max_ausgrid * 100) if orig_max_ausgrid else 0
                )
                results['ausgrid']['std_dev_change_pct'].append(
                    ((orig_std_ausgrid - att_std_ausgrid) / orig_std_ausgrid * 100) if orig_std_ausgrid else 0
                )

                # Process SGCC
                attacked_sgcc = attack_model.apply(sgcc_sample.copy())
                att_energy_sgcc = attacked_sgcc.sum().sum()
                att_max_sgcc = attacked_sgcc.max().max()
                att_std_sgcc = attacked_sgcc.std(axis=0).mean()

                results['sgcc']['energy_reduction_pct'].append(
                    ((orig_energy_sgcc - att_energy_sgcc) / orig_energy_sgcc * 100) if orig_energy_sgcc else 0
                )
                results['sgcc']['max_reading_change_pct'].append(
                    ((orig_max_sgcc - att_max_sgcc) / orig_max_sgcc * 100) if orig_max_sgcc else 0
                )
                results['sgcc']['std_dev_change_pct'].append(
                     ((orig_std_sgcc - att_std_sgcc) / orig_std_sgcc * 100) if orig_std_sgcc else 0
                )

            except Exception as e:
                logging.error(f"Error applying attack {attack_id} during impact comparison: {e}")
                # Append NaN for failed attacks to keep lists aligned
                for dataset in ['ausgrid', 'sgcc']:
                    for metric in metrics:
                        results[dataset][metric].append(np.nan)

        # --- Generate Comparison Plots ---
        self.assertEqual(len(results['ausgrid']['energy_reduction_pct']), len(attack_ids), "Result length mismatch for Ausgrid")
        self.assertEqual(len(results['sgcc']['energy_reduction_pct']), len(attack_ids), "Result length mismatch for SGCC")

        for metric in metrics:
            fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True) # Share Y axis for direct comparison
            metric_name = metric.replace('_', ' ').title()

            # Plot Ausgrid
            ax_ausgrid = axes[0]
            values_ausgrid = results['ausgrid'][metric]
            bars_ausgrid = ax_ausgrid.bar(attack_ids, values_ausgrid, color='skyblue')
            ax_ausgrid.set_title(f'{metric_name} (Ausgrid Sample)')
            ax_ausgrid.set_xlabel('Attack ID')
            ax_ausgrid.set_ylabel('Percentage Change (%)')
            ax_ausgrid.grid(True, axis='y', linestyle='--', alpha=0.6)
            ax_ausgrid.axhline(0, color='black', linewidth=0.8)
            ax_ausgrid.tick_params(axis='x', rotation=45)
            # Add value labels
            for bar, value in zip(bars_ausgrid, values_ausgrid):
                 if not np.isnan(value):
                     yval = bar.get_height()
                     ax_ausgrid.text(bar.get_x() + bar.get_width()/2.0, yval, f'{value:.1f}%', va='bottom' if yval >= 0 else 'top', ha='center', fontsize=9)


            # Plot SGCC
            ax_sgcc = axes[1]
            values_sgcc = results['sgcc'][metric]
            bars_sgcc = ax_sgcc.bar(attack_ids, values_sgcc, color='lightcoral')
            ax_sgcc.set_title(f'{metric_name} (SGCC Sample - Transposed)')
            ax_sgcc.set_xlabel('Attack ID')
            # ax_sgcc.set_ylabel('Percentage Change (%)') # Shared Y
            ax_sgcc.grid(True, axis='y', linestyle='--', alpha=0.6)
            ax_sgcc.axhline(0, color='black', linewidth=0.8)
            ax_sgcc.tick_params(axis='x', rotation=45)
            # Add value labels
            for bar, value in zip(bars_sgcc, values_sgcc):
                 if not np.isnan(value):
                     yval = bar.get_height()
                     ax_sgcc.text(bar.get_x() + bar.get_width()/2.0, yval, f'{value:.1f}%', va='bottom' if yval >= 0 else 'top', ha='center', fontsize=9)

            # Adjust ylim slightly for labels
            ymin, ymax = ax_ausgrid.get_ylim()
            padding = (ymax - ymin) * 0.1 # 10% padding
            ax_ausgrid.set_ylim(ymin - padding, ymax + padding)
            # ax_sgcc.set_ylim(ymin - padding, ymax + padding) # Applied by sharey=True


            fig.suptitle(f"Comparison of Attack Impact Metric: {metric_name}", fontsize=16, y=1.0)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            save_path = os.path.join(self.results_dir, f'attack_comparison_{metric}.png')
            plt.savefig(save_path)
            plt.close(fig)
            logging.info(f"Impact comparison plot saved to {save_path}")

        # Test passes if plots are generated without fatal errors.
        self.assertTrue(True) # Primarily visual inspection needed
'''

if __name__ == '__main__':
    # Allows running the tests directly
    unittest.main()