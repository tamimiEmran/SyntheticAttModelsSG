from __future__ import annotations
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
from src.data.sgcc_data_preparation import synthetic_set, real_set, make_cv_label_dataframe, sgcc_labels_path, combine_DatasetGroup,combine_DatasetGroup_leave_for_threshold
from src.models import catboost_model,svm_model, xgboost_model, knn_model, rf_model
from src.utils.evaluation import ExperimentResults, FoldResult, SplitResult, optimal_threshold, _default_metrics
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
    train: predictions | None = None
    val:   predictions | None = None
    test:  predictions | None = None

    # NEW ----------------------------------------------------------------
    # in ModelResult.as_fold() -------------------------------------------
    def as_fold(self, *, threshold: float | None = None) -> FoldResult:
        
        if threshold is None:
            threshold = optimal_threshold(self.val.y_true, self.val.y_pred)

        thr_metrics = _default_metrics(threshold=threshold)

        # validation split (still recorded, but no re-scan)
        val_split = SplitResult(
            y_true      = self.val.y_true,
            y_prob      = self.val.y_pred,
            metric_fns  = thr_metrics,           # ← just evaluate
            optimize_on = None                   # no scan
        )
        # train / test …
        train_split = SplitResult(
            y_true  = self.train.y_true,
            y_prob  = self.train.y_pred,
            metric_fns = thr_metrics
        )
        test_split  = SplitResult(
            y_true  = self.test.y_true,
            y_prob  = self.test.y_pred,
            metric_fns = thr_metrics
        )
        return FoldResult(train=train_split, val=val_split,
                        test=test_split,  train_time=self.train_time)
    

sgcc_labels = pd.read_hdf(sgcc_labels_path, key='original')
cv = make_cv_label_dataframe(labels = sgcc_labels, n_folds =10, random_state= 42)


all_results = {
    "synthetic": defaultdict(list),
    "real": defaultdict(list)
}

experiment = ExperimentResults()
model_name = "catboost"
for attack_type in ATTACK_TYPES_DEFAULT:
    for fold in range(1, 2):

        synthetic_results = ModelResult(name=f"syntheticResults_trained_on_{attack_type}_{model_name}_fold{fold}")
        real_results = ModelResult(name=f"realResults_trained_on_{attack_type}_{model_name}_fold{fold}")
        
        synthetic_dataset = synthetic_set(
            cv = cv,
            fold = fold,
            attackTypes = [attack_type],
            test_frac= 0.99
        )

        model = catboost_model.CatBoostModel()
        model_name = model.name
        training_time_start = time()
        model.fit(synthetic_dataset.train.X, synthetic_dataset.train.y)
        training_time_end = time()
        training_time = training_time_end - training_time_start
        synthetic_results.train_time = training_time

        synthetic_results.train = predictions(y_true=synthetic_dataset.train.y, y_pred=model.predict_proba(synthetic_dataset.train.X))
        synthetic_results.val = predictions(y_true=synthetic_dataset.val.y, y_pred=model.predict_proba(synthetic_dataset.val.X))    
        synthetic_results.test = predictions(y_true=synthetic_dataset.test.y, y_pred=model.predict_proba(synthetic_dataset.test.X))

        real_results.train_time = training_time
        real_dataset = real_set(cv=cv, fold= fold, random_state_base=42)
        real_dataset = combine_DatasetGroup_leave_for_threshold(real_dataset)
        real_results.train = predictions(y_true=real_dataset.test.y, y_pred=model.predict_proba(real_dataset.test.X))
        real_results.val = predictions(y_true=real_dataset.val.y, y_pred=model.predict_proba(real_dataset.val.X))
        real_results.test = predictions(y_true=real_dataset.test.y, y_pred=model.predict_proba(real_dataset.test.X))


        all_results["synthetic"][attack_type].append(cp(synthetic_results))
        all_results["real"][attack_type].append(cp(real_results))


        
        experiment.add_fold(
            attack_type = attack_type,
            fold_result_real= real_results.as_fold(),
            fold_result_syn = synthetic_results.as_fold()
        )

experiment.summary("synthetic", "attack_0")

