from __future__ import annotations
import os, sys, pickle, time
from collections import defaultdict
from copy import deepcopy as cp
from typing import Sequence, List, Dict

import numpy as np
import pandas as pd
from sklearn import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.sgcc_data_preparation import (
    synthetic_set, real_set, make_cv_label_dataframe,
    sgcc_labels_path, combine_DatasetGroup,
)
from src.models import catboost_model
from src.utils.evaluation import _default_metrics            # optional
from src.utils.evaluation import compute_optimal_threshold    # optional

from dataclasses import dataclass, field

# ─────────────────── constants ───────────────────────────────────────────
ATTACK_TYPES_DEFAULT: Sequence[str] = [f"attack_{i}" for i in range(13)] + ["attack_ieee"]
N_FOLDS = 10
RANDOM_STATE = 42
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "greedy_attack_removal")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────── light dataclasses for storage ───────────────────────
@dataclass
class Predictions:
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


# ─────────────────── helpers ─────────────────────────────────────────────
def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = metrics.roc_curve(y, p)
    return metrics.auc(fpr, tpr)


def train_on_fold(
    fold_id: int,
    attack_subset: Sequence[str],
    cv_labels: pd.DataFrame,
) -> FoldMetrics:
    """Train once on a single fold + subset, return AUC on *real* test."""
    # ---------------- build datasets (synthetic & real) ----------------
    synth_ds = synthetic_set(cv=cv_labels, fold=fold_id,
                             attackTypes=list(attack_subset), random_state_base= RANDOM_STATE)
    real_ds  = real_set(cv=cv_labels, fold=fold_id, random_state_base=RANDOM_STATE)
    real_ds  = combine_DatasetGroup(real_ds)        # train/val/test combined to be a single testset for the trained model on synthetic attacks

    # ---------------- model ----------------
    model = catboost_model.CatBoostModel(validationTuple=synth_ds.val)
    model.fit(synth_ds.train.X, synth_ds.train.y, verbose=False)

    # ---------------- test on real attacks (test split) ----------------
    y_true = real_ds.y
    y_prob = model.predict_proba(real_ds.X)[:, 1]

    return FoldMetrics(
        auc=auc_score(y_true, y_prob),
        cls_report=metrics.classification_report(
            y_true, y_prob > 0.5, output_dict=True
        )
    )


def evaluate_subset(
    attack_subset: Sequence[str],
    cv_labels: pd.DataFrame,
) -> SubsetResult:
    """Average real-world AUC over all folds for the given subset."""
    res = SubsetResult(subset=list(attack_subset))
    for fold in range(1, N_FOLDS + 1):
        res.fold_metrics.append(train_on_fold(fold, attack_subset, cv_labels))
    return res


# ─────────────────── greedy elimination loop ─────────────────────────────
def greedy_backward_elimination(
    attack_types: List[str],
    cv_labels: pd.DataFrame,
) -> List[SubsetResult]:
    """
    Returns a list of SubsetResult, one entry *per iteration*
    (baseline + every subsequent removal).
    """
    current_subset = attack_types.copy()
    history: List[SubsetResult] = []
    removed_order: List[str] = []          


    print(">>> Baseline: training on all synthetic attacks …")
    baseline = evaluate_subset(current_subset, cv_labels)
    history.append(baseline)
    print(f"    mean AUC = {baseline.mean_auc:.4f}")

    # iterate until only one attack remains
    while len(current_subset) > 1:
        best_auc = -np.inf
        best_to_drop = None
        best_result = None

        print(f"\n>>> Trying removals (current |S| = {len(current_subset)})")
        for atk in current_subset:
            candidate = [a for a in current_subset if a != atk]
            res = evaluate_subset(candidate, cv_labels)
            print(f"    drop {atk:>11s}:  mean AUC = {res.mean_auc:.4f}")
            if res.mean_auc > best_auc:
                best_auc, best_to_drop, best_result = res.mean_auc, atk, res

        # commit the removal
        print(f"--> Removing '{best_to_drop}'  (↑ best AUC = {best_auc:.4f})")
        current_subset.remove(best_to_drop)
        removed_order.append(best_to_drop)
        history.append(best_result)

        # persist intermediate state
        with open(os.path.join(RESULTS_DIR, "greedy_history.pkl"), "wb") as f:
            pickle.dump(history, f)

    return history, removed_order


# ─────────────────── main routine ────────────────────────────────────────
def main():
    print("Preparing cross-validation folds …")
    sgcc_labels = pd.read_hdf(sgcc_labels_path, key="original")
    cv_labels   = make_cv_label_dataframe(
        labels=sgcc_labels, n_folds=N_FOLDS, random_state=RANDOM_STATE
    )

    full_history, removed = greedy_backward_elimination(
        attack_types=list(ATTACK_TYPES_DEFAULT),
        cv_labels=cv_labels,
    )

    # save to .npy for the plotting helpers you already wrote
    np.save(os.path.join(RESULTS_DIR, "greedy_auc_per_subset.npy"),
            [ [m.auc for m in subset.fold_metrics] for subset in full_history ])
    np.save(os.path.join(RESULTS_DIR, "greedy_removed_order.npy"),
            removed)

    # pretty-print summary
    print("\n========== Summary ==========")

    for i, subset_res in enumerate(full_history):
        removed = "baseline" if i == 0 else f"-{ATTACK_TYPES_DEFAULT[i-1]}"
        print(f"{i:2d}. {removed:>10s}  |S|={len(subset_res.subset):2d}  "
              f"mean AUC = {subset_res.mean_auc:.4f}")
        

    for i, subset_res in enumerate(full_history):
        label = "baseline" if i == 0 else f"-{removed[i-1]}"
        print(f"{i:2d}. {label:>14s} |S|={len(subset_res.subset):2d}  "
               f"mean AUC = {subset_res.mean_auc:.4f}")

if __name__ == "__main__":
    main()
