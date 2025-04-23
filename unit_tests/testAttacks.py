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
        try:
            cls.sgcc_df, cls.sgcc_labels = load_sgcc_data()
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
        except ValueError as e:
            logging.error(f"Error loading SGCC data: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred loading SGCC data: {e}")

        # --- Load Ausgrid data ---
        try:
            cls.ausgrid_df = load_ausgrid_data()
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

if __name__ == '__main__':
    # Allows running the tests directly
    unittest.main()