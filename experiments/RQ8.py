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
# ❶ define the 5 learning algorithms you want to compare
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
m_name = "CatBoost"
model = MODEL_FACTORIES[m_name]()



#%%
#create numpy shape of (fold, train_attack_type, test_attack_type)
shape = (3, len(ATTACK_TYPES_DEFAULT), len(ATTACK_TYPES_DEFAULT))
results_arr = np.empty(shape, dtype=object)
# Load the dataset
for fold in range(1, 4):
    for train_attack_type in ATTACK_TYPES_DEFAULT:
        for test_attack_type in ATTACK_TYPES_DEFAULT:
        
            dataset = ausgrid_set(fold=fold, attackTypes= [], train_attackTypes = [train_attack_type], test_attackTypes=[test_attack_type] , random_state_base=42)

            m_name = "CatBoost"
            model = MODEL_FACTORIES[m_name]()
            model_name = model.name

            time_start = time()
            model.fit(dataset.train.X, dataset.train.y)
            time_end = time()

            training_time = time_end - time_start
            results = ModelResult(name=f"syntheticResults_trained_on_{m_name}_{model_name}_fold{fold}")
            results.train_time = training_time

            results.train = predictions(y_true=dataset.train.y, y_pred=model.predict_proba(dataset.train.X))
            results.val = predictions(y_true=dataset.val.y, y_pred=model.predict_proba(dataset.val.X))
            results.test = predictions(y_true=dataset.test.y, y_pred=model.predict_proba(dataset.test.X))

            results_arr[fold-1, ATTACK_TYPES_DEFAULT.index(train_attack_type), ATTACK_TYPES_DEFAULT.index(test_attack_type)] = cp(results)
            

        
    


