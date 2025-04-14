# experiments/tuning.py
"""
Hyperparameter Tuning using Optuna.

This script defines objective functions for different models and runs Optuna studies
to find optimal hyperparameters based on a validation dataset.
The results (best parameters) are saved for later use by experiments.
"""

import os
import sys
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import time
from tqdm import tqdm
import logging

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

# Import necessary components
from experiments import config
from experiments.data_preparation import prepare_experiment_data # To get validation data
# Import model classes directly for use within objective functions
from src.models import CatBoostModel, XGBoostModel, RandomForestModel, SVMModel, KNNModel
from src.utils.io import save_pickle # Use pickle for saving dicts

# --- Optuna Configuration ---
N_TRIALS = 100 # Number of Optuna trials (adjust as needed, original was 250)
N_SPLITS_CV = 3 # Number of cross-validation splits within the objective (original was 2)
STUDY_DIRECTION = "maximize" # Optimize for ROC AUC (maximize)
METRIC_TO_OPTIMIZE = "ROC-AUC"
OPTUNA_LOG_LEVEL = logging.WARNING # Reduce Optuna's verbosity

optuna.logging.set_verbosity(OPTUNA_LOG_LEVEL)

# --- Objective Functions ---

def _objective_catboost(trial: optuna.Trial, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Optuna objective function for CatBoost."""
    params = {
        # Search space adapted from original script
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5, log=True), # Adjusted range slightly
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 10.0, log=True), # Adjusted range slightly
        "border_count": trial.suggest_int("border_count", 32, 255), # Common range
        "random_strength": trial.suggest_float("random_strength", 1e-5, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0), # Range 0-1 usually
        # Fixed params
        "loss_function": "Logloss",
        "eval_metric": "AUC", # Optimize directly for AUC
        "task_type": "CPU", # Or "GPU" if available and configured
        "verbose": False,
        "random_seed": config.RANDOM_SEED,
    }

    kfold = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_SEED)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_val, y_val)):
        X_train_cv, X_test_cv = X_val[train_idx], X_val[test_idx]
        y_train_cv, y_test_cv = y_val[train_idx], y_val[test_idx]

        model = CatBoostModel(params=params)
        try:
            # Use eval_set for potential early stopping if iterations is large
            model.fit(X_train_cv, y_train_cv, eval_set=[(X_test_cv, y_test_cv)], early_stopping_rounds=50, verbose=False)
            y_proba = model.predict_proba(X_test_cv)[:, 1]
            score = roc_auc_score(y_test_cv, y_proba)
            scores.append(score)
        except Exception as e:
            print(f"Trial {trial.number}, Fold {fold+1} failed: {e}")
            return 0.0 # Return low score for failed trials

    return np.mean(scores)


def _objective_xgboost(trial: optuna.Trial, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Optuna objective function for XGBoost."""
    params = {
        # Search space adapted from original script
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12), # Adjusted min depth
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0), # Adjusted min subsample
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0), # Adjusted min colsample
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
        # Fixed params
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "use_label_encoder": False,
        "verbosity": 0,
        "random_state": config.RANDOM_SEED,
        "tree_method": "hist" # Often faster for larger datasets
    }

    kfold = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_SEED)
    scores = []
    y_val_int = y_val.astype(int) # Ensure labels are integers

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_val, y_val_int)):
        X_train_cv, X_test_cv = X_val[train_idx], X_val[test_idx]
        y_train_cv, y_test_cv = y_val_int[train_idx], y_val_int[test_idx]

        model = XGBoostModel(params=params)
        try:
            model.fit(X_train_cv, y_train_cv, eval_set=[(X_test_cv, y_test_cv)], early_stopping_rounds=50, verbose=False)
            y_proba = model.predict_proba(X_test_cv)[:, 1]
            score = roc_auc_score(y_test_cv, y_proba)
            scores.append(score)
        except Exception as e:
            print(f"Trial {trial.number}, Fold {fold+1} failed: {e}")
            return 0.0

    return np.mean(scores)

def _objective_rf(trial: optuna.Trial, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Optuna objective function for RandomForest."""
    params = {
        # Search space adapted from original script
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50), # Steps can speed up search
        "max_depth": trial.suggest_int("max_depth", 5, 30), # None is also an option
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20), # Wider range
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20), # Wider range
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]), # Consider class weight
        # Fixed params
        "random_state": config.RANDOM_SEED,
        "n_jobs": -1
    }

    kfold = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_SEED)
    scores = []
    y_val_ravel = y_val.ravel() # RF needs 1D y

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_val, y_val_ravel)):
        X_train_cv, X_test_cv = X_val[train_idx], X_val[test_idx]
        y_train_cv, y_test_cv = y_val_ravel[train_idx], y_val_ravel[test_idx]

        model = RandomForestModel(params=params)
        try:
            model.fit(X_train_cv, y_train_cv)
            y_proba = model.predict_proba(X_test_cv)[:, 1]
            score = roc_auc_score(y_test_cv, y_proba)
            scores.append(score)
        except Exception as e:
            print(f"Trial {trial.number}, Fold {fold+1} failed: {e}")
            return 0.0

    return np.mean(scores)

def _objective_svm(trial: optuna.Trial, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Optuna objective function for SVM."""
    params = {
        # Search space adapted from original script
        "C": trial.suggest_float("C", 1e-4, 1e3, log=True), # Wider C range
        "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]), # Add common kernels
        "degree": trial.suggest_int("degree", 2, 5), # Relevant only for 'poly'
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        "coef0": trial.suggest_float("coef0", 0.0, 1.0), # Relevant for 'poly', 'sigmoid'
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        # Fixed params
        "probability": True, # Must be True for predict_proba and ROC AUC
        "random_state": config.RANDOM_SEED,
    }
    # Ensure degree is only used if kernel is poly
    if params["kernel"] != "poly":
        del params["degree"]
    # Ensure coef0 is only used if kernel is poly or sigmoid
    if params["kernel"] not in ["poly", "sigmoid"]:
         del params["coef0"]


    kfold = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_SEED)
    scores = []
    y_val_ravel = y_val.ravel()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_val, y_val_ravel)):
        X_train_cv, X_test_cv = X_val[train_idx], X_val[test_idx]
        y_train_cv, y_test_cv = y_val_ravel[train_idx], y_val_ravel[test_idx]

        # SVM can be sensitive to scale - ensure data is scaled *before* tuning
        # Assuming X_val is already scaled appropriately by the caller
        model = SVMModel(params=params)
        try:
            model.fit(X_train_cv, y_train_cv)
            y_proba = model.predict_proba(X_test_cv)[:, 1]
            score = roc_auc_score(y_test_cv, y_proba)
            scores.append(score)
        except Exception as e:
            print(f"Trial {trial.number}, Fold {fold+1} failed: {e}")
            return 0.0

    return np.mean(scores)

def _objective_knn(trial: optuna.Trial, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Optuna objective function for KNN."""
    params = {
        # Search space adapted from original script
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 50), # Wider range
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "leaf_size": trial.suggest_int("leaf_size", 10, 50),
        "p": trial.suggest_int("p", 1, 2), # 1=Manhattan, 2=Euclidean
        # Fixed params
        'n_jobs': -1
    }

    kfold = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_SEED)
    scores = []
    y_val_ravel = y_val.ravel()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_val, y_val_ravel)):
        X_train_cv, X_test_cv = X_val[train_idx], X_val[test_idx]
        y_train_cv, y_test_cv = y_val_ravel[train_idx], y_val_ravel[test_idx]

        # KNN is sensitive to scale - ensure data is scaled *before* tuning
        model = KNNModel(params=params)
        try:
            model.fit(X_train_cv, y_train_cv)
            y_proba = model.predict_proba(X_test_cv)[:, 1]
            score = roc_auc_score(y_test_cv, y_proba)
            scores.append(score)
        except Exception as e:
            print(f"Trial {trial.number}, Fold {fold+1} failed: {e}")
            return 0.0

    return np.mean(scores)


# --- Runner Function ---

_OBJECTIVE_MAP = {
    'catboost': _objective_catboost,
    'xgboost': _objective_xgboost,
    'rf': _objective_rf,
    'svm': _objective_svm,
    'knn': _objective_knn,
}

def run_tuning(
    model_name: str,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = N_TRIALS
) -> Dict[str, Any]:
    """
    Runs the Optuna hyperparameter tuning study for a given model.

    Args:
        model_name (str): Name of the model to tune (e.g., 'catboost').
        X_val (np.ndarray): Validation features. **Should be appropriately scaled** if model requires it (SVM, KNN).
        y_val (np.ndarray): Validation labels.
        n_trials (int): Number of Optuna trials to run.

    Returns:
        Dict[str, Any]: Dictionary containing the best hyperparameters found.

    Raises:
        ValueError: If model_name is not supported.
    """
    if model_name not in _OBJECTIVE_MAP:
        raise ValueError(f"Unsupported model for tuning: {model_name}. Supported: {list(_OBJECTIVE_MAP.keys())}")

    objective_func = _OBJECTIVE_MAP[model_name]

    # --- TQDM Callback for Optuna ---
    pbar = tqdm(total=n_trials, desc=f"Tuning {model_name}")
    def tqdm_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        pbar.update(1)
        if study.best_trial:
            pbar.set_postfix_str(f"Best {METRIC_TO_OPTIMIZE}: {study.best_value:.4f}", refresh=True)
    # --------------------------------

    print(f"\nStarting Optuna tuning for {model_name} ({n_trials} trials)...")
    study = optuna.create_study(direction=STUDY_DIRECTION)
    study.optimize(
        lambda trial: objective_func(trial, X_val, y_val),
        n_trials=n_trials,
        callbacks=[tqdm_callback],
        gc_after_trial=True # Garbage collect after each trial
    )
    pbar.close()

    print(f"Tuning complete for {model_name}.")
    print(f"  Best Trial Number: {study.best_trial.number}")
    print(f"  Best {METRIC_TO_OPTIMIZE}: {study.best_value:.6f}")
    print(f"  Best Hyperparameters: {study.best_params}")

    return study.best_params


# --- Main Execution (Example Usage) ---
if __name__ == "__main__":
    print("Running Hyperparameter Tuning Script...")
    set_seed(config.RANDOM_SEED)

    # --- 1. Prepare Validation Data ---
    # This requires deciding which data to tune on. Let's assume we tune
    # separately for 'real' and 'synthetic' scenarios using fold 1 data
    # as the base validation set. Ideally, use cross-validation across
    # multiple folds for more robust tuning, but that takes much longer.

    # Example: Prepare validation set from Fold 1 'real' data
    print("\nPreparing validation data (Example: Fold 1, Real)...")
    # This uses the *full* fold 1 training data as the basis for tuning.
    # It does NOT use the test set of fold 1.
    # A common practice is to split the training data of a fold further.
    # Let's use prepare_experiment_data to get fold 1 train/test, then use the
    # *training* part of that fold as our tuning validation set.

    # Choose data type ('real' or 'synthetic') and fold for tuning
    TUNING_DATA_TYPE = 'real' # Tune parameters for models trained on real data
    TUNING_FOLD_ID = 1

    # Define configs to get the desired data split using prepare_experiment_data
    # We only need the training part of the fold split for tuning.
    train_cfg = {'type': TUNING_DATA_TYPE, 'oversample': None} # No oversampling during tuning itself
    test_cfg = {'type': TUNING_DATA_TYPE} # Dummy test config, we only use the train output

    try:
        (X_tune, y_tune), _ = prepare_experiment_data(
            fold_id=TUNING_FOLD_ID,
            train_config=train_cfg,
            test_config=test_cfg # Test set output is ignored here
        )
        print(f"Validation data prepared. Shape: {X_tune.shape}")

        # **Important:** Scale validation data if tuning SVM or KNN
        if any(m in ['svm', 'knn'] for m in config.MODEL_LIST):
             print("Scaling validation data (StandardScaler)...")
             from sklearn.preprocessing import StandardScaler
             scaler = StandardScaler()
             # Reshape if needed (assuming examples are [n_samples, n_features])
             original_shape = X_tune.shape
             X_tune_reshaped = X_tune.reshape(-1, X_tune.shape[-1]) # Flatten if needed
             X_tune_scaled_reshaped = scaler.fit_transform(X_tune_reshaped)
             X_tune = X_tune_scaled_reshaped.reshape(original_shape)
             print("Scaling complete.")


    except Exception as e:
        print(f"Error preparing validation data: {e}. Exiting.")
        sys.exit(1)


    # --- 2. Run Tuning for Each Model ---
    all_best_params = {}
    for model_to_tune in config.MODEL_LIST:
        try:
            best_params = run_tuning(model_to_tune, X_tune, y_tune)
            all_best_params[model_to_tune] = best_params

            # --- 3. Save Best Parameters ---
            # Save using the convention expected by config.load_hyperparameters
            save_filename = f"{model_to_tune.lower()}_parameters_{TUNING_DATA_TYPE}.npy"
            save_filepath = os.path.join(config.HYPERPARAMS_DIR, save_filename)
            print(f"Saving best parameters for {model_to_tune} ({TUNING_DATA_TYPE}) to {save_filepath}...")
            # Save as numpy object array containing the dict
            np.save(save_filepath, best_params)
            # Or use pickle via utils.io
            # save_pickle(best_params, save_filepath.replace('.npy', '.pkl'))
            print("Parameters saved.")

        except Exception as e:
            print(f"\n!!! Error during tuning or saving for model {model_to_tune}: {e} !!!")
            import traceback
            traceback.print_exc()


    print("\nHyperparameter tuning script finished.")
    print("Saved parameters:", all_best_params)
