# scripts/save_attacked_ausgrid.py
"""
Loads the raw Ausgrid dataset from 'data/raw/ausgrid/', applies various synthetic
attacks month-by-month using the attack models from src/, and saves the original
and attacked dataframes to 'data/processed/ausgrid_attacked.h5'.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
import time
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Adjust sys.path to find the 'src' directory ---
# Assumes the script is run from the project root (e.g., 'python scripts/save_attacked_ausgrid.py')
# Correctly finds the project root relative to the script's location
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    logging.debug(f"Project Root: {PROJECT_ROOT}")
    logging.debug(f"SRC Directory added to path: {SRC_DIR}")
except NameError:
    logging.error("Could not determine project root automatically. Please ensure script is run correctly.")
    sys.exit(1)
# ---------------------------------------------------

# Now import from src modules
try:
    # Assuming loader is now in src.data.loader
    from data.loader import load_ausgrid_data
    from attack_models import get_attack_model, list_available_attacks
except ImportError as e:
    logging.error(f"Failed to import necessary modules from src: {e}")
    logging.error(f"Ensure the script is run from the project root and SRC_DIR '{SRC_DIR}' is correct.")
    sys.exit(1)

# --- Configuration ---
# Define paths relative to the project root
RAW_DATA_BASE_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'ausgrid') # Base Ausgrid raw data directory
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_FILENAME = 'ausgrid_attacked.h5'
OUTPUT_FILEPATH = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILENAME)

# --- !! IMPORTANT !! ---
# Directory names inside data/raw/ausgrid/ containing the CSVs
AUSGRID_SUBDIRS = [
    'ausgrid2010', # e.g., data/raw/ausgrid/ausgrid2010/ausgrid2010.csv
    'ausgrid2011',
    'ausgrid2012'
]
AUSGRID_FILENAMES = [
    'ausgrid2010.csv',
    'ausgrid2011.csv',
    'ausgrid2012.csv'
]
# --- Make sure these match your actual structure ---

# Construct full paths to the subdirectories containing the CSVs
AUSGRID_DIR_PATHS = []
for subdir in AUSGRID_SUBDIRS:
    path = os.path.join(RAW_DATA_BASE_DIR, subdir)
    if not os.path.isdir(path):
        logging.warning(f"Ausgrid subdirectory not found: {path}. Skipping.")
    else:
        AUSGRID_DIR_PATHS.append(path)

if not AUSGRID_DIR_PATHS:
    logging.error(f"No valid Ausgrid subdirectories found in {RAW_DATA_BASE_DIR} based on AUSGRID_SUBDIRS.")
    logging.error("Please check your data structure and configuration.")
    sys.exit(1)

AUSGRID_DATE_FORMATS = [0, 1, 1] # Corresponds to filenames/years [2010, 2011, 2012]

# Define which attacks to generate and save
try:
    # ATTACK_IDS_TO_PROCESS = list_available_attacks() # Process all available
    ATTACK_IDS_TO_PROCESS = [*[str(i) for i in range(13)], 'ieee'] # Explicit list matching original code
    logging.info(f"Attacks to process: {ATTACK_IDS_TO_PROCESS}")
except Exception as e:
    logging.error(f"Could not retrieve available attacks: {e}")
    sys.exit(1)


# --- Main Script Logic ---
def main():
    """Loads data, applies attacks, and saves results."""
    script_start_time = time.time()
    logging.info("Starting script: save_attacked_ausgrid.py")
    logging.info(f"Output Path: {OUTPUT_FILEPATH}")

    # --- 1. Load Original Data ---
    logging.info("Loading Ausgrid data...")
    try:
        # Pass the list of full directory paths and corresponding filenames/formats
        original_df = load_ausgrid_data(
            dir_paths=AUSGRID_DIR_PATHS,
            filenames=AUSGRID_FILENAMES[:len(AUSGRID_DIR_PATHS)], # Match filenames to valid paths
            date_formats=AUSGRID_DATE_FORMATS[:len(AUSGRID_DIR_PATHS)] # Match formats to valid paths
        )
        logging.info(f"Ausgrid data loaded successfully. Shape: {original_df.shape}")
        if original_df.empty:
             logging.error("Loaded Ausgrid DataFrame is empty. Exiting.")
             sys.exit(1)
        if not isinstance(original_df.index, pd.DatetimeIndex):
            logging.warning("Index is not DatetimeIndex. Attempting conversion...")
            original_df.index = pd.to_datetime(original_df.index, errors='coerce')
            if original_df.index.isna().any():
                logging.error("Failed to convert index to DatetimeIndex. Exiting.")
                sys.exit(1)

    except FileNotFoundError as e:
        logging.error(f"Error loading Ausgrid data: {e}")
        logging.error(f"Please ensure CSV files exist in the specified subdirectories within {RAW_DATA_BASE_DIR}.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        sys.exit(1)


    # --- 2. Ensure Output Directory Exists ---
    try:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        logging.info(f"Output directory '{PROCESSED_DATA_DIR}' ensured.")
    except OSError as e:
        logging.error(f"Failed to create output directory '{PROCESSED_DATA_DIR}': {e}")
        sys.exit(1)

    # --- 3. Save Original Data ---
    logging.info(f"Saving original data to {OUTPUT_FILENAME} (key='original')...")
    try:
        original_df.to_hdf(
            OUTPUT_FILEPATH,
            key='original',
            mode='w', # Write mode for the first save
            format='table',
            complib='blosc',
            complevel=5
        )
        logging.info("Original data saved.")
    except Exception as e:
        logging.error(f"Error saving original data: {e}", exc_info=True)
        # Continue to process attacks even if saving original fails? Optional.
        # sys.exit(1)


    # --- 4. Apply and Save Attacks ---
    logging.info("Applying and saving attacks...")

    for attack_id in ATTACK_IDS_TO_PROCESS:
        attack_start_time = time.time()
        logging.info(f"Processing Attack ID: {attack_id}")
        try:
            # Get the attack model instance
            attack_model = get_attack_model(attack_id)
            logging.info(f"  - Using model: {attack_model.__class__.__name__}")

            # Group original data by month
            try:
                 monthly_groups = original_df.groupby(original_df.index.to_period('M'))
            except Exception as e:
                 logging.error(f"Error grouping DataFrame by month: {e}. Ensure index is datetime.", exc_info=True)
                 continue # Skip this attack if grouping fails


            modified_monthly_dfs = []
            logging.info("  - Applying attack month by month...")
            # Use tqdm for progress bar
            group_iterator = tqdm(monthly_groups, desc=f"  Attack {attack_id}", leave=False, unit="month", ncols=100)
            for group_name, group_data in group_iterator:
                if group_data.empty:
                    continue # Skip empty months

                # Apply the attack to the current month's data
                modified_month = attack_model.apply(group_data.copy()) # Pass a copy

                # Basic validation
                if modified_month.shape != group_data.shape:
                     logging.warning(f"Shape mismatch for month {group_name} after attack {attack_id}. Original: {group_data.shape}, Modified: {modified_month.shape}. Skipping month.")
                     continue
                if not modified_month.index.equals(group_data.index):
                     logging.warning(f"Index mismatch for month {group_name} after attack {attack_id}. Aligning index.")
                     modified_month = modified_month.reindex(group_data.index) # Attempt to fix index

                modified_monthly_dfs.append(modified_month)

            if not modified_monthly_dfs:
                logging.warning(f"No monthly data generated for attack {attack_id}. Skipping save.")
                continue

            # Concatenate the modified monthly dataframes
            logging.info("  - Concatenating monthly results...")
            attacked_df = pd.concat(modified_monthly_dfs)
            attacked_df = attacked_df.sort_index() # Ensure chronological order

            # Final validation (optional but recommended)
            if not attacked_df.index.equals(original_df.index):
                 logging.warning(f"Final index of attacked data does not match original for attack {attack_id}. Reindexing.")
                 attacked_df = attacked_df.reindex(original_df.index)


            # Save the attacked dataframe
            # Use consistent key naming, e.g., 'attack_0', 'attack_ieee'
            save_key = f'attack_{attack_id}'
            logging.info(f"  - Saving attacked data (key='{save_key}')...")
            attacked_df.to_hdf(
                OUTPUT_FILEPATH,
                key=save_key,
                mode='a',       # Append mode for subsequent saves
                format='table',
                complib='blosc',
                complevel=5
            )
            attack_duration = time.time() - attack_start_time
            logging.info(f"  - Attack {attack_id} processed and saved in {attack_duration:.2f} seconds.")

        except ValueError as e:
            logging.error(f"Error getting/using attack model for ID {attack_id}: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"An unexpected error occurred processing attack {attack_id}: {e}", exc_info=True)

    total_duration = time.time() - script_start_time
    logging.info(f"Script finished in {total_duration:.2f} seconds.")
    logging.info(f"Attacked data saved to: {OUTPUT_FILEPATH}")

if __name__ == "__main__":
    # Ensure the script can find necessary packages and the src directory
    logging.info(f"Current sys.path: {sys.path}")
    main()
