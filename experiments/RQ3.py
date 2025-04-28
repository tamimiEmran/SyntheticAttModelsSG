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
from src.data.sgcc_data_preparation import synthetic_set, real_set, make_cv_label_dataframe, sgcc_labels_path, combine_DatasetGroup
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

sgcc_labels = pd.read_hdf(sgcc_labels_path, key='original')
cv = make_cv_label_dataframe(labels = sgcc_labels, n_folds =10, random_state= 42)
real_dataset = real_set(cv=cv, fold=1, random_state=42)
real_dataset = combine_DatasetGroup(real_dataset)

all_results = {
    "synthetic": defaultdict(list),
    "real": defaultdict(list)
}

for attack_type in ATTACK_TYPES_DEFAULT:
    for fold in range(1, 10):
        model = catboost_model.CatBoostModel()
        model_name = model.name
        synthetic_results = ModelResult(name=f"syntheticResults_trained_on_{attack_type}_{model_name}_fold{fold}")
        real_results = ModelResult(name=f"realResults_trained_on_{attack_type}_{model_name}_fold{fold}")
        #
        synthetic_dataset = synthetic_set(
            cv = cv,
            fold = fold,
            attack_type = [attack_type]
        )
        
        training_time_start = time()
        model.fit(synthetic_dataset.train.X, synthetic_dataset.train.y)
        training_time_end = time()
        training_time = training_time_end - training_time_start
        synthetic_results.train_time = training_time

        synthetic_results.train = predictions(y_true=synthetic_dataset.train.y, y_pred=model.predict_proba(synthetic_dataset.train.X))
        synthetic_results.val = predictions(y_true=synthetic_dataset.val.y, y_pred=model.predict_proba(synthetic_dataset.val.X))    
        synthetic_results.test = predictions(y_true=synthetic_dataset.test.y, y_pred=model.predict_proba(synthetic_dataset.test.X))

        real_results.train_time = training_time
        real_results.train = predictions(y_true=real_dataset.train.y, y_pred=model.predict_proba(real_dataset.train.X))
        real_results.val = predictions(y_true=real_dataset.val.y, y_pred=model.predict_proba(real_dataset.val.X))
        real_results.test = predictions(y_true=real_dataset.test.y, y_pred=model.predict_proba(real_dataset.test.X))


        all_results["synthetic"][attack_type].append(cp(synthetic_results))
        all_results["real"][attack_type].append(cp(real_results))


        




