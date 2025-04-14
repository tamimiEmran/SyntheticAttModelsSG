# experiments/data_preparation.py
"""
Provides functions to prepare training and testing datasets for experiments
based on configuration settings (fold, attack types, real/synthetic).
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
from src.data.preprocessing import apply_pca_transformation, apply_adasyn, apply_smote # Add oversampling imports if needed


# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

from experiments import config # Experiment configurations
from src.data.preprocessing import create_monthly_examples, apply_adasyn, filter_excessive_zeros # Or daily
from src.data.fold_generator import get_fold
from src.attack_models import get_attack_model

# --- Global Data Cache (Load once) ---
# It's often more efficient to load base data once if experiments run in sequence
# or are managed by a higher-level script. Otherwise, load within the function.
_BASE_DATA_CACHE = {
    "original_df": None,
    "labels_series": None,
    "real_theft_ids": None,
    "all_benign_ids": None,
}

def _load_base_data_once():
    """Loads base data into the cache if not already loaded."""
    if _BASE_DATA_CACHE["original_df"] is None:
        print("Loading base data for preparation...")
        try:
            df = pd.read_hdf(config.ORIGINAL_H5_FILE, key='df')
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                df.index = pd.to_datetime(df.index)
            _BASE_DATA_CACHE["original_df"] = df

            labels = pd.read_hdf(config.ORIGINAL_LABELS_H5_FILE, key='df')
            if isinstance(labels, pd.DataFrame):
                labels = labels.squeeze()
            _BASE_DATA_CACHE["labels_series"] = labels

            _BASE_DATA_CACHE["real_theft_ids"] = labels[labels == 1].index.tolist()
            _BASE_DATA_CACHE["all_benign_ids"] = labels[labels == 0].index.tolist()
            print("Base data loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: Required base data file not found: {e}")
            raise # Re-raise to stop execution if base data is missing
        except Exception as e:
            print(f"Error loading base data: {e}")
            raise


def _apply_attacks_to_consumers(
    base_df: pd.DataFrame,
    consumer_ids: List[str],
    attack_ids: List[str]
) -> pd.DataFrame:
    """
    Applies specified attacks to the given consumers, distributing consumers
    among attacks if multiple attack IDs are provided.

    Returns a DataFrame containing ONLY the attacked consumers' data.
    """
    if not consumer_ids or not attack_ids:
        return pd.DataFrame(index=base_df.index) # Return empty DF

    df_to_attack = base_df[consumer_ids].copy()
    attacked_df_combined = df_to_attack.copy() # Start with original data

    # Distribute consumers among attacks
    consumer_splits = np.array_split(consumer_ids, len(attack_ids))

    print(f"    - Applying attacks {attack_ids} to {len(consumer_ids)} consumers...")
    for i, current_attack_id in enumerate(attack_ids):
        ids_for_this_attack = consumer_splits[i]
        if not len(ids_for_this_attack): continue # Skip if split is empty

        print(f"      - Applying {current_attack_id} to {len(ids_for_this_attack)} consumers...")
        attack_model = get_attack_model(current_attack_id)
        target_df_part = df_to_attack[ids_for_this_attack]

        attacked_parts_list = []
        try:
            # Apply monthly
            monthly_groups = target_df_part.groupby(target_df_part.index.to_period('M'))
            for _, group_data in monthly_groups:
                if not group_data.empty:
                    attacked_parts_list.append(attack_model.apply(group_data))

            if attacked_parts_list:
                attacked_df_part = pd.concat(attacked_parts_list).sort_index()
                # Ensure index/columns match before updating
                attacked_df_part = attacked_df_part.reindex(target_df_part.index)
                attacked_df_part = attacked_df_part[target_df_part.columns]
                # Update the combined DataFrame only for these consumers
                attacked_df_combined.update(attacked_df_part)
            else:
                 print(f"      - Warning: No data generated for attack {current_attack_id} on its consumer subset.")


        except Exception as e:
            print(f"      - Error applying attack {current_attack_id}: {e}")
            # Decide how to handle: keep original data for these consumers or skip?
            # Keeping original data by not updating attacked_df_combined for these IDs.

    return attacked_df_combined


def prepare_experiment_data(
    fold_id: int,
    train_config: Dict[str, Any],
    test_config: Dict[str, Any]
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Prepares scaled train and test data arrays based on configuration.

    Args:
        fold_id (int): The fold number (1 to N_FOLDS).
        train_config (Dict): Configuration for training data. Keys include:
            'type': 'synthetic' or 'real'.
            'attack_ids': List of attack IDs if type is 'synthetic'.
            'oversample': 'adasyn', 'smote', or None.
            'split_ratio': Ratio for splitting benign consumers for attack targets (e.g., 0.5). Default 0.5.
        test_config (Dict): Configuration for testing data. Keys include:
            'type': 'synthetic' or 'real'.
            'attack_ids': List of attack IDs if type is 'synthetic'.

    Returns:
        Tuple containing:
            - (X_train, y_train): Prepared training data and labels.
            - (X_test, y_test): Prepared testing data and labels.
    """
    _load_base_data_once() # Ensure base data is loaded

    # Retrieve cached base data
    base_df = _BASE_DATA_CACHE["original_df"]
    all_labels = _BASE_DATA_CACHE["labels_series"]
    real_theft_ids = _BASE_DATA_CACHE["real_theft_ids"]
    all_benign_ids = _BASE_DATA_CACHE["all_benign_ids"]

    print(f"Preparing Data - Fold: {fold_id}, Train: {train_config['type']}, Test: {test_config['type']}")

    # Get benign consumer IDs for this fold's train/test split
    benign_train_ids, benign_test_ids = get_fold(fold_id, config.FOLDS_FILE)

    X_train, y_train, X_test, y_test = None, None, None, None

    # --- Prepare Training Data ---
    print("  - Preparing Training Data...")
    train_type = train_config['type']
    oversample_method = train_config.get('oversample', None)
    synth_split_ratio = train_config.get('split_ratio', 0.5)

    if train_type == 'synthetic':
        attack_ids = train_config.get('attack_ids', [])
        if not attack_ids: raise ValueError("attack_ids must be provided for synthetic training data")

        # Split benign training consumers for attack application
        if len(benign_train_ids) < 2: # Need at least 2 to split
             raise ValueError(f"Not enough benign consumers ({len(benign_train_ids)}) in fold {fold_id} training set to create synthetic data.")
        synth_honest_ids, synth_attack_target_ids = train_test_split(
            benign_train_ids, test_size=synth_split_ratio, random_state=config.RANDOM_SEED
        )

        # Apply attacks
        attacked_consumers_df = _apply_attacks_to_consumers(
            base_df, synth_attack_target_ids, attack_ids
        )

        # Combine honest portion and attacked portion
        train_df_combined = pd.concat([
            base_df[synth_honest_ids],
            attacked_consumers_df # Contains only the attacked consumers
        ], axis=1)

        # Create labels (0 for honest, 1 for attacked)
        train_labels = pd.Series(0, index=synth_honest_ids + synth_attack_target_ids)
        train_labels.loc[synth_attack_target_ids] = 1

        # Create examples
        X_train_raw, y_train_raw = create_monthly_examples(train_df_combined, train_labels)

    elif train_type == 'real':
        # Use benign training IDs + all real theft IDs
        train_consumer_ids = benign_train_ids + real_theft_ids
        train_df_combined = base_df[train_consumer_ids]
        train_labels = all_labels.loc[train_consumer_ids]

        # Create examples
        X_train_raw, y_train_raw = create_monthly_examples(train_df_combined, train_labels)

    else:
        raise ValueError(f"Invalid train_config type: {train_type}")

    print(f"    Raw train examples: {X_train_raw.shape}")

    # Apply oversampling if configured
    if oversample_method == 'adasyn':
        print(f"    Applying ADASYN oversampling...")
        X_train, y_train = apply_adasyn(X_train_raw, y_train_raw, random_state=config.RANDOM_SEED)
        print(f"    Train examples after ADASYN: {X_train.shape}")
    elif oversample_method == 'smote':
         # X_train, y_train = apply_smote(X_train_raw, y_train_raw, random_state=config.RANDOM_SEED)
         print("SMOTE not fully implemented here yet, using raw data.") # Placeholder
         X_train, y_train = X_train_raw, y_train_raw # Use raw if not implemented
    else:
        X_train, y_train = X_train_raw, y_train_raw

    # --- Prepare Testing Data ---
    print("  - Preparing Testing Data...")
    test_type = test_config['type']

    if test_type == 'synthetic':
        attack_ids = test_config.get('attack_ids', [])
        if not attack_ids: raise ValueError("attack_ids must be provided for synthetic testing data")

        # Use benign test consumers for generating synthetic test data
        if len(benign_test_ids) < 2:
            print(f"Warning: Not enough benign consumers ({len(benign_test_ids)}) in fold {fold_id} test set for synthetic testing. Test set might be small or empty.")
            # Handle this case - maybe use training consumers? Or skip? For now, proceed.
            test_honest_ids, test_attack_target_ids = benign_test_ids, [] # No attacks if only 1
            if len(benign_test_ids) == 1:
                 test_honest_ids, test_attack_target_ids = benign_test_ids, []
            else: # len == 0
                 test_honest_ids, test_attack_target_ids = [], []
        else:
             test_honest_ids, test_attack_target_ids = train_test_split(
                 benign_test_ids, test_size=0.5, random_state=config.RANDOM_SEED # Use same split ratio logic?
             )


        # Apply attacks
        attacked_consumers_df_test = _apply_attacks_to_consumers(
            base_df, test_attack_target_ids, attack_ids
        )

        # Combine
        test_df_combined = pd.concat([
            base_df[test_honest_ids],
            attacked_consumers_df_test
        ], axis=1)

        # Create labels
        test_labels = pd.Series(0, index=test_honest_ids + test_attack_target_ids)
        test_labels.loc[test_attack_target_ids] = 1

        # Create examples
        X_test, y_test = create_monthly_examples(test_df_combined, test_labels)

    elif test_type == 'real':
        # Use benign test IDs + all real theft IDs
        test_consumer_ids = benign_test_ids + real_theft_ids
        test_df_combined = base_df[test_consumer_ids]
        test_labels = all_labels.loc[test_consumer_ids]

        # Create examples
        X_test, y_test = create_monthly_examples(test_df_combined, test_labels)

    else:
        raise ValueError(f"Invalid test_config type: {test_type}")

    print(f"    Test examples: {X_test.shape}")

    # Optional: Filter zeros (apply to both train and test for consistency if needed, or just test)
    # X_test, y_test = filter_excessive_zeros(X_test, y_test, max_zeros=10)
    # print(f"    Test examples after zero filtering: {X_test.shape}")

    # --- Final Checks ---
    if X_train is None or y_train is None or X_test is None or y_test is None:
        raise RuntimeError("Data preparation failed to produce valid train/test arrays.")
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        raise RuntimeError("Mismatch between example and label counts after preparation.")
    if X_train.shape[0] == 0:
        print(f"Warning: Training set for fold {fold_id} is empty.")
    if X_test.shape[0] == 0:
        print(f"Warning: Test set for fold {fold_id} is empty.")


    # Note: Scaling is not included here, assuming it might be part of a pipeline
    # or applied within the runner if needed. Could be added here too.

    return (X_train, y_train), (X_test, y_test)



# experiments/data_preparation.py
def prepare_pca_data_for_experiment(
    fold_id: int,
    data_config: Dict[str, Any],
    pca_variance_threshold: float = 0.90
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Prepares PCA-transformed train and test data arrays for experiments.

    Applies PCA to the base dataset first, then splits consumers based on the
    fold and data_config ('real' or 'synthetic'). The features become the
    principal components for each selected consumer.

    Args:
        fold_id (int): The fold number (1 to N_FOLDS).
        data_config (Dict): Configuration for data selection. Keys include:
            'type': 'real' or 'synthetic'.
            'oversample': 'adasyn', 'smote', or None (applied to train set).
        pca_variance_threshold (float): Variance threshold for PCA.

    Returns:
        Tuple containing:
            - (X_train, y_train): Prepared training data (PCA features) and labels.
            - (X_test, y_test): Prepared testing data (PCA features) and labels.
    """
    _load_base_data_once() # Ensure base data is loaded

    # Retrieve cached base data
    base_df = _BASE_DATA_CACHE["original_df"]
    all_labels = _BASE_DATA_CACHE["labels_series"]
    real_theft_ids = _BASE_DATA_CACHE["real_theft_ids"]
    all_benign_ids = _BASE_DATA_CACHE["all_benign_ids"]

    print(f"Preparing PCA Data - Fold: {fold_id}, Config: {data_config}, PCA Var: {pca_variance_threshold}")

    # 1. Apply PCA Transformation to the entire base dataset
    # Note: PCA is applied *before* fold splitting or synthetic/real selection
    pca_transformed_data = apply_pca_transformation(
        base_df,
        pca_variance_threshold=pca_variance_threshold,
        standardize=True # Usually recommended
    )

    if pca_transformed_data.empty:
        raise RuntimeError("PCA transformation failed or produced empty results.")

    # 2. Get consumer IDs for this fold's train/test split
    benign_train_ids, benign_test_ids = get_fold(fold_id, config.FOLDS_FILE)

    # 3. Determine train/test consumer IDs based on config type
    data_type = data_config['type']
    oversample_method = data_config.get('oversample', None)

    if data_type == 'synthetic':
        # For PCA, 'synthetic' usually means selecting consumers based on
        # the fold split, but using their *original* PCA profiles.
        # The concept of applying attacks *after* PCA doesn't make sense.
        # So, 'synthetic' and 'real' configs primarily affect *which* consumers
        # are selected for train/test based on the fold split of benign IDs.
        # Let's assume we train on the benign train split and test on benign test split + real theft.
        # This interpretation might need adjustment based on the original intent of PCA_fullDataExamples.
        print("  PCA Data Prep: Using 'synthetic' config for consumer selection (benign train vs benign test + real).")
        train_consumer_ids = benign_train_ids # Train only on benign consumers from train split
        test_consumer_ids = benign_test_ids + real_theft_ids # Test on benign test split + real theft

    elif data_type == 'real':
         train_consumer_ids = benign_train_ids + real_theft_ids # Train on benign train split + real theft
         test_consumer_ids = benign_test_ids + real_theft_ids # Test on benign test split + real theft
         # Note: This leads to overlap if just splitting this way.
         # The original PCA_fullDataExamples used train_test_split on the *combined* list
         # of positive/negative consumers defined by the fold. Let's replicate that.

         print("  PCA Data Prep: Using 'real' config for consumer selection (train/test split on fold consumers).")
         # Combine consumers relevant to this fold type ('real' uses benign + real theft)
         positive_ids = real_theft_ids
         # Use ALL benign consumers for the split pool if 'real' config implies using all available data
         # Or just the fold's benign consumers? Let's use fold's consumers like original.
         negative_ids = benign_train_ids + benign_test_ids
         all_fold_consumers = positive_ids + negative_ids
         all_fold_labels = all_labels.loc[all_fold_consumers]

         # Split these consumers into train/test
         train_consumer_ids, test_consumer_ids, _, _ = train_test_split(
             all_fold_consumers,
             all_fold_labels,
             test_size=0.3, # Standard test split size? Adjust if needed.
             random_state=config.RANDOM_SEED,
             stratify=all_fold_labels
         )

    else:
        raise ValueError(f"Invalid data_config type: {data_type}")

    # 4. Select PCA features and labels for train/test sets
    print(f"  Selecting PCA features for {len(train_consumer_ids)} train / {len(test_consumer_ids)} test consumers...")
    X_train = pca_transformed_data.loc[train_consumer_ids].values
    y_train = all_labels.loc[train_consumer_ids].values

    X_test = pca_transformed_data.loc[test_consumer_ids].values
    y_test = all_labels.loc[test_consumer_ids].values

    # 5. Apply oversampling to training data if configured
    if oversample_method == 'adasyn':
        print(f"    Applying ADASYN oversampling to PCA training data...")
        X_train, y_train = apply_adasyn(X_train, y_train, random_state=config.RANDOM_SEED)
        print(f"    Train examples after ADASYN: {X_train.shape}")
    elif oversample_method == 'smote':
        print(f"    Applying SMOTE oversampling to PCA training data...")
        # X_train, y_train = apply_smote(X_train, y_train, random_state=config.RANDOM_SEED) # Assuming apply_smote exists
        print("    SMOTE application placeholder.") # Placeholder
    else:
         pass # No oversampling

    print(f"  PCA Data Preparation Complete: Train shape={X_train.shape}, Test shape={X_test.shape}")
    return (X_train, y_train), (X_test, y_test)


