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
    m_name: defaultdict(list) for m_name in MODEL_FACTORIES.keys()
}

#%%
# Load the dataset
for fold in range(1, 10):
    dataset = ausgrid_set(fold=fold, attackTypes = ATTACK_TYPES_DEFAULT , random_state_base=42)
    for m_name, factory in MODEL_FACTORIES.items():
        model = factory()
        model_name = model.name
        results = ModelResult(name=f"syntheticResults_trained_on_{m_name}_{model_name}_fold{fold}")

        start = time()
        model.fit(dataset.train.X, dataset.train.y)
        end = time()

        training_time = end - start
        results.train_time = training_time

        results.train = predictions(y_true=dataset.train.y, y_pred=model.predict_proba(dataset.train.X))
        results.val = predictions(y_true=dataset.val.y, y_pred=model.predict_proba(dataset.val.X))
        results.test = predictions(y_true=dataset.test.y, y_pred=model.predict_proba(dataset.test.X))

        all_results[m_name][fold] = cp(results)
        print(f"Model {m_name} trained in {training_time:.2f} seconds")


