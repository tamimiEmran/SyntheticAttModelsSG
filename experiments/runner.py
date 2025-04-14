# experiments/runner.py
"""
Provides functions to run a single experimental trial (e.g., one fold).
"""
import os
import sys
import numpy as np
from typing import Dict, Any, Tuple

# --- Add project root to sys.path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------

from experiments import config
from experiments.data_preparation import prepare_experiment_data
# Import specific model classes needed, or use a factory if preferred
from src.models import CatBoostModel, XGBoostModel, RandomForestModel, SVMModel, KNNModel
from src.evaluation.metrics import calculate_metrics

# Map model names to classes
MODEL_CLASSES = {
    'catboost': CatBoostModel,
    'xgboost': XGBoostModel,
    'rf': RandomForestModel,
    'svm': SVMModel,
    'knn': KNNModel,
}

def run_single_trial(
    fold_id: int,
    model_name: str,
    train_config: Dict[str, Any],
    test_config: Dict[str, Any],
    hyperparams: Optional[Dict[str, Any]] = None,
    load_params_if_none: bool = True
) -> Dict[str, float]:
    """
    Runs a single trial: prepares data, trains model, evaluates, returns metrics.

    Args:
        fold_id (int): The fold number.
        model_name (str): Name of the model (e.g., 'catboost', 'rf').
        train_config (Dict): Configuration for training data preparation.
        test_config (Dict): Configuration for testing data preparation.
        hyperparams (Optional[Dict]): Pre-defined hyperparameters. If None and
                                      load_params_if_none is True, attempts to load
                                      them using config.load_hyperparameters.
        load_params_if_none (bool): Whether to try loading params if none are provided.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics for this trial.
                          Returns empty dict if trial fails.
    """
    print(f"\n--- Running Trial: Fold={fold_id}, Model={model_name}, Train={train_config['type']}, Test={test_config['type']} ---")
    metrics = {}
    try:
        # 1. Prepare Data
        (X_train, y_train), (X_test, y_test) = prepare_experiment_data(
            fold_id, train_config, test_config
        )

        # Handle empty data case
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
             print("  Skipping trial due to empty train or test set.")
             return {} # Return empty metrics for failed trial

        # 2. Load/Set Hyperparameters
        if hyperparams is None and load_params_if_none:
            # Determine data type key for loading params (e.g., 'real' or 'synthetic')
            param_data_type = train_config['type']
            print(f"  Loading hyperparameters for {model_name} ({param_data_type})...")
            hyperparams = config.load_hyperparameters(model_name, param_data_type)
        elif hyperparams is None:
            print("  Using default hyperparameters.")
            hyperparams = {} # Use model defaults

        # 3. Initialize Model
        model_class = MODEL_CLASSES.get(model_name)
        if not model_class:
            raise ValueError(f"Unknown model name: {model_name}")
        model = model_class(params=hyperparams)
        print(f"  Initialized {model_name} with params: {model.get_params()}")

        # 4. Train Model
        # Add eval_set for models that support it, if validation data were prepared
        # For simplicity now, just fit on train
        model.fit(X_train, y_train)
        print("  Model training complete.")

        # 5. Evaluate Model
        print("  Evaluating model...")
        y_pred = model.predict(X_test)
        y_proba = None
        # Need predict_proba for AUC, handle models that might not have it (like basic SVM without probability=True)
        try:
            if hasattr(model.model, 'predict_proba'):
                 y_proba = model.predict_proba(X_test)[:, 1] # Probability of positive class
            # Handle SVM decision function as alternative for ranking if needed
            elif hasattr(model.model, 'decision_function') and model_name == 'svm':
                 y_proba = model.decision_function(X_test) # Use decision values for AUC calculation
                 print("  Using SVM decision_function for ROC-AUC calculation.")

        except AttributeError as e:
             print(f"  Warning: predict_proba not available or failed for {model_name}. AUC will be None. Error: {e}")


        metrics = calculate_metrics(y_test, y_pred, y_proba)
        print(f"  Evaluation Metrics: {metrics}")

    except Exception as e:
        print(f"!! Error during trial (Fold {fold_id}, Model {model_name}): {e}")
        # Optionally log the full traceback for debugging
        # import traceback
        # traceback.print_exc()
        return {} # Return empty metrics for failed trial

    return metrics
