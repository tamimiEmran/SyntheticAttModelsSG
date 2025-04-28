import os
import sys
import numpy as np
import pickle
from collections import defaultdict
from dataclasses import dataclass
#add project root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Imports ---
from src.data.ausgrid_data_preparation import ausgrid_set
from src.models import catboost_model,svm_model, xgboost_model, knn_model, rf_model
from src.utils.evaluation import evaluate_aggregated, evaluate_monthly, _default_metrics, compute_optimal_threshold
from time import time
from copy import deepcopy as cp
from typing import Callable, Dict, Sequence
import pandas as pd

ATTACK_TYPES_DEFAULT: Sequence[str] = [f"attack_{i}" for i in range(13) ] + ["attack_ieee"]

@dataclass
class predictions:
    y_true: np.ndarray = None
    y_pred: np.ndarray = None


@dataclass
class ModelResult:
    name: str
    train_time: float = 0.0
    train: predictions = None
    val: predictions = None
    test: predictions = None


MODEL_FACTORIES = dict(
    CatBoost = catboost_model.CatBoostModel,
    SVM      = svm_model.SVMModel,
    XGBoost  = xgboost_model.XGBoostModel,
    kNN      = knn_model.KNNModel,
    RF       = rf_model.RandomForestModel,
)

all_results = {
    f"fold_{fold}": defaultdict(list) for fold in range(1, 10)
}


N_FOLDS = 3
ATTACK_TYPES = [f"attack_{i}" for i in range(13)] + ["attack_ieee"]
BASE_RANDOM_STATE = 42
MODEL_NAME = "CatBoost"


def train_and_eval(fold, train_attacks, test_attacks):
    """Return test-set AUC for one fold."""
    ds = ausgrid_set(
        fold            = fold,
        n_folds         = N_FOLDS,
        attackTypes     = train_attacks,        
        train_attackTypes = train_attacks,
        test_attackTypes  = test_attacks,
        random_state_base = BASE_RANDOM_STATE,
    )
    model = MODEL_FACTORIES[MODEL_NAME]()
    model.fit(ds.train.X, ds.train.y)
    y_pred = model.predict_proba(ds.test.X)
    auc    = evaluate_aggregated(ds.test.y, y_pred)["auc-roc"]
    return auc

def cv_mean_auc(train_attacks, test_attacks):
    return np.mean([
        train_and_eval(fold, train_attacks, test_attacks)
        for fold in range(1, N_FOLDS + 1)
    ])

current_subset   = []          
remaining        = ATTACK_TYPES.copy()
history          = []          

while remaining:
    best_auc   = -np.inf
    best_attack = None

    
    for att in remaining:
        candidate = current_subset + [att]
        auc = cv_mean_auc(candidate, ATTACK_TYPES)   
        if auc > best_auc:
            best_auc, best_attack = auc, att

    
    current_subset.append(best_attack)
    remaining.remove(best_attack)
    history.append((current_subset.copy(), best_auc))
    print(f"Added {best_attack:<11} ➜ CV-AUC = {best_auc:.4f}")


print("\nGreedy-addition trajectory:")
for i, (subset, auc) in enumerate(history, 1):
    print(f"{i:2d}: {subset}  →  AUC {auc:.4f}")
