from __future__ import annotations
import os, sys, pickle, time
from collections import defaultdict
from copy import deepcopy as cp
from typing import Sequence, List, Dict

import numpy as np
import pandas as pd
from sklearn import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
from src.data.sgcc_data_preparation import (
    synthetic_set, real_set, make_cv_label_dataframe,
    sgcc_labels_path)
from src.models import catboost_model, svm_model, xgboost_model, knn_model, rf_model
from dataclasses import dataclass, field

# ─────────────────── constants ───────────────────────────────────────────
ATTACK_TYPES_DEFAULT: Sequence[str] = [f"attack_{i}" for i in range(13)] + ["attack_ieee"]
N_FOLDS = 10
RANDOM_STATE = 42
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "greedy_attack_removal")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────── light dataclasses for storage ───────────────────────
@dataclass
class predictions:
    y_true: np.ndarray
    y_pred: np.ndarray       # probability for class 1


@dataclass
class FoldMetrics:
    auc: float
    cls_report: Dict[str, float]


@dataclass
class SubsetResult:                        # one attack subset across all folds
    subset: List[str]
    fold_metrics: List[FoldMetrics] = field(default_factory=list)

    @property
    def mean_auc(self) -> float:
        return float(np.mean([m.auc for m in self.fold_metrics]))


@dataclass
class ModelResult:
    """Container for all outputs from one training / evaluation run."""
    name: str
    train_time: float = 0.0
    train: predictions = None
    val: predictions = None
    test: predictions = None

    # ──────────────── convenience helpers ───────────────────────────────
    def auc_scores(self) -> Dict[str, float]:
        """Return ROC-AUC for each split."""
        return {
            "train": metrics.roc_auc_score(self.train.y_true, self.train.y_pred),
            "val":   metrics.roc_auc_score(self.val.y_true,   self.val.y_pred),
            "test":  metrics.roc_auc_score(self.test.y_true,  self.test.y_pred),
        }

    @property
    def mean_auc(self) -> float:
        """Average AUC across train / val / test splits."""
        scores = self.auc_scores().values()
        return float(np.mean(list(scores)))
# ─────────────────── RQ-5: algorithm-ranking experiment ──────────────────
from src.models import (
    catboost_model, svm_model, xgboost_model, knn_model, rf_model
)

# ❶ define the 5 learning algorithms you want to compare
MODEL_FACTORIES = dict(
    CatBoost = catboost_model.CatBoostModel,
    SVM      = svm_model.SVMModel,
    XGBoost  = xgboost_model.XGBoostModel,
    kNN      = knn_model.KNNModel,
    RF       = rf_model.RandomForestModel,
)

# ❷ storage scaffold:  method → "train/test" scenario → list[fold] of ModelResult
rq5_results = {
    m_name: defaultdict(list)   # keys will be: "syn→syn", "syn→real", "real→real"
    for m_name in MODEL_FACTORIES
}

# ❸ prepare labels & fold splitter only once
sgcc_labels = pd.read_hdf(sgcc_labels_path, key="original")
cv          = make_cv_label_dataframe(labels=sgcc_labels, n_folds=10,
                                      random_state=42)

# ❹ loop over folds and models
for fold in range(1, 11):
    # --- full synthetic and full real datasets for THIS fold -------------
    syn_ds  = synthetic_set(cv=cv, fold=fold, attackTypes=ATTACK_TYPES_DEFAULT, random_state_base=42)
    real_ds = real_set(cv=cv, fold=fold, random_state_base=42)
    

    for m_name, factory in MODEL_FACTORIES.items():
        # -------- a) train + test on synthetic ---------------------------
        model_syn = factory( validationTuple=syn_ds.val)
        t0 = time.time()
        model_syn.fit(syn_ds.train.X, syn_ds.train.y)
        train_time = time.time() - t0

        res_syn_syn = ModelResult(
            name=f"{m_name}_syn→syn_fold{fold}", train_time=train_time,
            train=predictions(syn_ds.train.y,
                              model_syn.predict_proba(syn_ds.train.X)),
            val=predictions(syn_ds.val.y,
                            model_syn.predict_proba(syn_ds.val.X)),
            test=predictions(syn_ds.test.y,
                             model_syn.predict_proba(syn_ds.test.X)),
        )
        rq5_results[m_name]["syn→syn"].append(cp(res_syn_syn))

        # -------- b) syn-trained model tested on real --------------------
        res_syn_real = ModelResult(
            name=f"{m_name}_syn→real_fold{fold}", train_time=train_time,
            train=predictions(syn_ds.train.y,
                              model_syn.predict_proba(syn_ds.train.X)),
            val=predictions(syn_ds.val.y,
                            model_syn.predict_proba(syn_ds.val.X)),
            test=predictions(real_ds.test.y,
                             model_syn.predict_proba(real_ds.test.X))
        )
        rq5_results[m_name]["syn→real"].append(cp(res_syn_real))

        # -------- c) train + test on real --------------------------------
        model_real = factory()
        t0 = time.time()
        model_real.fit(real_ds.train.X, real_ds.train.y)
        train_time_real = time.time() - t0

        res_real_real = ModelResult(
            name=f"{m_name}_real→real_fold{fold}", train_time=train_time_real,
            train=predictions(real_ds.train.y,
                              model_real.predict_proba(real_ds.train.X)),
            val=predictions(real_ds.val.y,
                            model_real.predict_proba(real_ds.val.X)),
            test=predictions(real_ds.test.y,
                             model_real.predict_proba(real_ds.test.X)),
        )
        rq5_results[m_name]["real→real"].append(cp(res_real_real))

# ❺ (optional) persist everything for later plotting / stats -------------
os.makedirs("results/rq5_algorithm_ranking", exist_ok=True)
with open("results/rq5_algorithm_ranking/all_results.pkl", "wb") as f:
    pickle.dump(rq5_results, f)

print("RQ-5 experiment complete — results saved to "
      "`results/rq5_algorithm_ranking/all_results.pkl`")
