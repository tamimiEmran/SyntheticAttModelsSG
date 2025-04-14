# experiments/config.py
"""
Central configuration settings for experiments.
"""

import os
import numpy as np

# --- Paths ---
# Assumes the script using this config is in experiments/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results') # Directory to save experiment outputs
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models') # Directory to save trained models (optional)
FOLDS_FILE = os.path.join(PROCESSED_DATA_DIR, '10folds.pkl') # Example path for saved folds

# Ensure directories exist (optional, scripts can also ensure this)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) # Ensure processed data dir exists

# --- Data Files ---
SGCC_DATA_FILE = os.path.join(RAW_DATA_DIR, 'sgcc_data.csv') # Replace with actual filename
AUSGRID_DIRS = [
    # Replace with actual paths if Ausgrid is used directly by experiments
    # os.path.join(RAW_DATA_DIR, 'ausgrid2010'),
    # os.path.join(RAW_DATA_DIR, 'ausgrid2011'),
    # os.path.join(RAW_DATA_DIR, 'ausgrid2012'),
]
AUSGRID_ATTACKED_FILE = os.path.join(PROCESSED_DATA_DIR, 'ausgrid_attacked.h5')
# Original SGCC data file used in applyAttacks.py - adjust if needed
ORIGINAL_H5_FILE = os.path.join(PROCESSED_DATA_DIR, 'original.h5')
ORIGINAL_LABELS_H5_FILE = os.path.join(PROCESSED_DATA_DIR, 'original_labels.h5')


# --- Experiment Settings ---
RANDOM_SEED = 42
N_FOLDS = 10 # Number of cross-validation folds commonly used

# --- Model Settings ---
MODEL_LIST = ['catboost', 'xgboost', 'rf', 'svm', 'knn'] # Consistent naming

# Define where to find pre-tuned hyperparameters (if loading)
# Adjust filenames as per your old saving convention
HYPERPARAMS_DIR = os.path.join(PROJECT_ROOT, 'hyperparameters') # Example directory
os.makedirs(HYPERPARAMS_DIR, exist_ok=True)

def load_hyperparameters(model_name: str, data_type: str) -> dict:
    """Loads hyperparameters from .npy file."""
    # Example filenames: 'catboost_parameters_real.npy', 'rf_parameters_synthetic.npy'
    filename = f"{model_name.lower()}_parameters_{data_type}.npy"
    filepath = os.path.join(HYPERPARAMS_DIR, filename)
    default_params = {} # Define default params if file not found?
    if os.path.exists(filepath):
        try:
            # Ensure allow_pickle=True if they were saved as objects/dicts
            params = np.load(filepath, allow_pickle=True).item()
            print(f"Loaded hyperparameters for {model_name} ({data_type}) from {filepath}")
            return params
        except Exception as e:
            print(f"Warning: Could not load hyperparameters from {filepath}. Error: {e}. Using defaults.")
            return default_params
    else:
        print(f"Warning: Hyperparameter file not found: {filepath}. Using defaults.")
        return default_params

# --- Attack Settings ---
# List of attack IDs used in experiments
# Ensure consistency with attack_models/implementations.py and factory.py
# Using strings for flexibility (e.g., 'ieee')
ATTACK_IDS_ALL = [*[str(i) for i in range(13)], 'ieee']


# --- Evaluation Settings ---
DEFAULT_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']


ATTACK_CONSTANTS = {
    "attack1range": [0.1, 0.8],         # Range for multiplicative noise/scaling (Attacks 1, 3, 6, 7)
    "AttackReading": 7,                # Window size in readings (used in Attacks 2, 4, 8, 11). Note: Original was 1*7. Seems small if daily readings=48? Verify context.
    "i_max": [0.3, 1.0],               # Range for max reduction factor (Attack 8)
    "attack11Factor": 0.3,             # Reduction factor for peak window (Attack 11) - implies 70% theft?
    "attack4": [0.1, 0.9],             # Range for multiplicative factor in window (Attack 4)
    "attack5": [0.1, 0.9],             # Range for factor applied to mean (Attack 5)
    "rocAttIntense": [0.05, 0.1],      # Range for ramp intensity (Attack 8)
    "attack12Factor": 0.4,             # Factor for comparing sums (Attack 12) - Note: Original was 1.1 in placeholder description, actual is 0.4
}
