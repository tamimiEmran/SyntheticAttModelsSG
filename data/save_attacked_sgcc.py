# scripts/save_attacked_sgcc.py
"""
Loads the preprocessed SGCC dataset, applies various synthetic attacks
month-by-month using the attack models, and saves the original and
attacked dataframes to an HDF5 file specifically for SGCC data.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
import time
import numpy as np # Import numpy

# --- Adjust sys.path to find the 'src' directory ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ---------------------------------------------------

# Use config for paths and attack lists
from experiments import config
from src.data.loader import load_sgcc_data
from src.attack_models import get_attack_model, list_available_attacks

# --- Configuration ---
# Input SGCC raw data file path from config
SGCC_RAW_FILE = config.SGCC_DATA_FILE # Ensure this is set correctly in config.py

# Output directory and filename for the attacked SGCC HDF5 file
OUTPUT_DIR = config.PROCESSED_DATA_DIR
OUTPUT_FILENAME = 'sgcc_attacked.h5' # Specific filename for SGCC
OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Define which attacks to generate and save (usually all)
ATTACK_IDS_TO_PROCESS = config.ATTACK_IDS_ALL

# --- Main Script Logic ---
def main():
    """Loads SGCC data, applies attacks, and saves results."""
    start_time = time.time()

    print("Starting script: save_attacked_sgcc.py")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Output Path: {OUTPUT_FILEPATH}")

    # --- 1. Load and Prepare Original SGCC Data ---
    print(f"\nLoading and preparing SGCC data from: {SGCC_RAW_FILE}...")
    try:
        # load_sgcc_data returns features_df (Consumers x Time), labels_series
        # We need Time x Consumers for the attack application logic used previously.
        # Let's adjust the loading or transpose after loading.
        consumers_df, labels_series = load_sgcc_data(SGCC_RAW_FILE)

        # Transpose consumers_df to get Time x Consumers format
        original_df = consumers_df.T
        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(original_df.index):
            original_df.index = pd.to_datetime(original_df.index)
        original_df = original_df.sort_index()

        print(f"SGCC data loaded and prepared successfully. Shape: {original_df.shape}")
        if original_df.empty:
             print("Error: Loaded SGCC DataFrame is empty. Exiting.")
             sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error loading SGCC data: {e}")
        print(f"Please ensure the file exists at {SGCC_RAW_FILE}.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during SGCC data loading/preparation: {e}")
        sys.exit(1)


    # --- 2. Ensure Output Directory Exists ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' ensured.")

    # --- 3. Save Original Data ---
    # Save the Time x Consumers DataFrame
    print(f"\nSaving original SGCC data to {OUTPUT_FILENAME} (key='original')...")
    try:
        original_df.to_hdf(OUTPUT_FILEPATH, key='original', mode='w', format='table')
        print("Original SGCC data saved.")
        # Optionally save labels if needed alongside (though usually handled separately)
        # labels_series.to_hdf(OUTPUT_FILEPATH, key='original_labels', mode='a', format='table')
    except Exception as e:
        print(f"Error saving original SGCC data: {e}")
        # Decide whether to continue or exit
        # sys.exit(1)


    # --- 4. Apply and Save Attacks ---
    print(f"\nApplying and saving attacks: {ATTACK_IDS_TO_PROCESS}")

    # Define consumers to apply attacks to (e.g., only benign ones?)
    # For pre-computation, often applied to *all* consumers, and selection happens later.
    # Let's apply to all consumers present in original_df.
    consumers_to_process = original_df.columns

    for attack_id in ATTACK_IDS_TO_PROCESS:
        attack_start_time = time.time()
        print(f"\nProcessing Attack ID: {attack_id}")
        try:
            # Get the attack model instance
            attack_model = get_attack_model(attack_id)
            print(f"  - Using model: {attack_model.__class__.__name__}")

            # --- Apply attack month-by-month ---
            # Group original data by month
            try:
                 monthly_groups = original_df.groupby(original_df.index.to_period('M'))
            except Exception as e:
                 print(f"Error grouping DataFrame by month: {e}. Ensure index is datetime.")
                 continue # Skip this attack if grouping fails

            modified_monthly_dfs = []
            print("  - Applying attack month by month...")
            for group_name, group_data in tqdm(monthly_groups, desc=f"  Attack {attack_id}", leave=False):
                if group_data.empty:
                    continue # Skip empty months

                # Select only the consumers we intend to process (all in this case)
                group_data_subset = group_data[consumers_to_process]

                # Apply the attack to the current month's data subset
                modified_month = attack_model.apply(group_data_subset.copy()) # Pass a copy

                # Basic validation
                if modified_month.shape != group_data_subset.shape:
                     print(f"Warning: Shape mismatch for month {group_name} after attack {attack_id}. Skipping month.")
                     continue
                if not modified_month.index.equals(group_data_subset.index):
                     print(f"Warning: Index mismatch for month {group_name} after attack {attack_id}. Aligning index.")
                     modified_month = modified_month.reindex(group_data_subset.index) # Attempt to fix index

                modified_monthly_dfs.append(modified_month)

            if not modified_monthly_dfs:
                print(f"Warning: No monthly data generated for attack {attack_id}. Skipping save.")
                continue

            # Concatenate the modified monthly dataframes
            print("  - Concatenating monthly results...")
            attacked_df = pd.concat(modified_monthly_dfs)
            attacked_df = attacked_df.sort_index() # Ensure chronological order

            # Final validation (optional) - Ensure it aligns with original index
            if not attacked_df.index.equals(original_df.index):
                 print(f"Warning: Final index of attacked data does not match original for attack {attack_id}.")
                 # Attempt reindexing, though this might hide issues
                 attacked_df = attacked_df.reindex(original_df.index)

            # Save the attacked dataframe
            save_key = f'attack_{attack_id}' # Consistent key naming
            print(f"  - Saving attacked data (key='{save_key}')...")
            attacked_df.to_hdf(OUTPUT_FILEPATH, key=save_key, mode='a', format='table') # Append mode
            attack_duration = time.time() - attack_start_time
            print(f"  - Attack {attack_id} processed and saved in {attack_duration:.2f} seconds.")

        except ValueError as e:
            print(f"Error getting attack model for ID {attack_id}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred processing attack {attack_id}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging

    total_duration = time.time() - start_time
    print(f"\nScript finished in {total_duration:.2f} seconds.")
    print(f"Attacked SGCC data saved to: {OUTPUT_FILEPATH}")

if __name__ == "__main__":
    main()
