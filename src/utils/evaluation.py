from __future__ import annotations
import numpy as np
import pandas as pd
import numpy as np
from typing import Iterable, Tuple, Dict
from typing import Callable, Dict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

def return_metrics_results(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    metrics: Dict[str, Callable] | None = None,
    optimizeOn: tuple[np.ndarray, np.ndarray] | None = None
) -> Dict[str, float]:
    """
    Returns a dictionary of metrics for the given true and predicted values.
    If `metrics` is None, it will use the default metrics.
    optimizeOn is a tuple of (y_true, y_pred_prob) to optimize the threshold on the f1 score on.
    expect that y_pred could be a probability vector of shape (n_samples, n_classes) if so take the second column. 

    """


    if optimizeOn is not None:
        y_true_val, y_pred_prob_val = optimizeOn
        if y_pred_prob_val.ndim == 2 and y_pred_prob_val.shape[1] > 1:
            y_pred_prob_val = y_pred_prob_val[:, 1]
        threshold = optimal_threshold(
            y_true_val, y_pred_prob_val, metric="f1", return_metrics=True
        )[0]

        metrics = _default_metrics(threshold=threshold)

    if metrics is None:
        metrics = _default_metrics()



    # Check if y_pred_prob is a 2D array and take the second column if necessary
    if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] > 1:
        y_pred_prob = y_pred_prob[:, 1]

    

    return {name: fn(y_true, y_pred_prob) for name, fn in metrics.items()}
    


def optimal_threshold(
    y_true:     np.ndarray,
    y_pred_prob: np.ndarray,
    *,
    metric: str = "f1",
    thresholds: Iterable[float] | None = None,
    return_metrics: bool = False,
) -> float | Tuple[float, Dict[str, float]]:
    
    if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] > 1:
        y_pred_prob = y_pred_prob[:, 1]

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 10_001)

    best_thr     = 0.5
    best_val     = -np.inf
    metric_funcs = dict(
        f1        = lambda y, p: f1_score(y, p, zero_division=0),
        accuracy  = accuracy_score,
        precision = lambda y, p: precision_score(y, p, zero_division=0),
        recall    = lambda y, p: recall_score(y, p, zero_division=0),
    )
    if metric not in metric_funcs:
        raise ValueError(
            f"`metric` must be one of {list(metric_funcs)}; got {metric!r}"
        )

    for thr in thresholds:
        preds  = y_pred_prob >= thr
        val    = metric_funcs[metric](y_true, preds)
        if val > best_val:
            best_val, best_thr = val, thr

    if not return_metrics:
        return float(best_thr)

    final_preds = y_pred_prob >= best_thr
    metrics_dict = dict(
        threshold = float(best_thr),
        f1        = f1_score(y_true, final_preds, zero_division=0),
        accuracy  = accuracy_score(y_true, final_preds),
        precision = precision_score(y_true, final_preds, zero_division=0),
        recall    = recall_score(y_true, final_preds, zero_division=0),
    )
    return best_thr, metrics_dict
def compute_optimal_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    *,
    metric: str = "f1",
) -> float:

    return optimal_threshold(y_val, p_val, metric=metric)

def pad_and_stats_vec(month_vals: np.ndarray, days_per_example: int = 31):
    """
    month_vals : 2-D array (n_cons * months, ≤31)

    Returns
    -------
    X : 2-D array (n_cons * months, 4 + days_per_example)
    """
    n_cons, n_days = month_vals.shape
    X = np.zeros((n_cons, 4 + days_per_example), month_vals.dtype)

    X[:, 0] = month_vals.mean(1)
    X[:, 1] = month_vals.std(1)
    X[:, 2] = month_vals.min(1)
    X[:, 3] = month_vals.max(1)
    X[:, 4 : 4 + n_days] = month_vals      # copy only the real days

    return X


def predict_consumer_month_probs(
    model,
    df: pd.DataFrame,
    *,
    days_per_example: int = 31,
    positive_class: int = 1,
    dtype=np.float32,
):
    """
    Vectorised, single-batch version of `predict_consumer_month_probs`.
    """
    periods = df.columns.to_period("M")
    unique_periods = periods.unique()
    
    X_parts      = []
    index_parts  = []

    # Pre-compute once to avoid repeated Series.__getitem__ in the loop
    col_periods_arr = periods.to_numpy()

    for period in unique_periods:
        cols_mask   = (col_periods_arr == period)          
        month_vals  = df.iloc[:, cols_mask].to_numpy(
            dtype=dtype, copy=False
        )                                                  # shape (n_cons, ≤31)
        if month_vals.size == 0:
            continue

        X_month = pad_and_stats_vec(month_vals, days_per_example)

        X_parts.append(X_month)
        index_parts.extend(zip(df.index, [period] * len(df)))

    
    X = np.vstack(X_parts)
    probs = model.predict_proba(X)[:, positive_class]
    idx = pd.MultiIndex.from_tuples(index_parts, names=["consumer_id", "period"])
    return pd.Series(probs, index=idx, name="probability").sort_index()





def _default_metrics(
    *,                                     
    threshold: float = 0.50                 
) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:

    def bin_wrap(fn_bin):
        thr = float(threshold)
        return lambda y, p: fn_bin(y, p >= thr)

    return dict(
        roc_auc       = roc_auc_score,
        avg_precision = average_precision_score,
        accuracy      = bin_wrap(accuracy_score),
        f1            = bin_wrap(lambda y, b: f1_score(y, b, zero_division=0)),
        precision     = bin_wrap(lambda y, b: precision_score(y, b, zero_division=0)),
        recall        = bin_wrap(lambda y, b: recall_score(y, b, zero_division=0)),
        log_loss      = lambda y, p: log_loss(y, np.c_[1 - p, p], labels=[0, 1]),
    )

# ------------------------------------------------------------------
# 1.  Evaluate directly on consumer-month predictions
# ------------------------------------------------------------------
def evaluate_monthly(
    model,
    df: pd.DataFrame,
    labels: pd.Series,
    *,
    validation_threshold: float = None,
    metrics: Dict[str, Callable] | None = None,
    days_per_example: int = 31,
    positive_class: int = 1,
    dtype=np.float32,
) -> Dict[str, float]:
    """
    Scores `model` on every (consumer, month) probability it outputs.

    Returns
    -------
    A dict {metric_name: score}.
    """
    # get probabilities (MultiIndex Series)
    probs = predict_consumer_month_probs(
        model, df,
        days_per_example=days_per_example,
        positive_class=positive_class,
        dtype=dtype,
    )

    # replicate each consumer's label for each month
    y_true = labels.reindex(
        probs.index.get_level_values("consumer_id")
    ).to_numpy()

    if metrics is None:
        metrics = _default_metrics()

    return {name: fn(y_true, probs.to_numpy()) for name, fn in metrics.items()}


def evaluate_aggregated(
    model,
    df: pd.DataFrame,
    labels: pd.Series,
    *,
    validation_threshold: float = None,
    agg: str | Callable = "mean",          # mean, median, np.max, …
    metrics: Dict[str, Callable] | None = None,
    days_per_example: int = 31,
    positive_class: int = 1,
    dtype=np.float32,
) -> Dict[str, float]:
    """
    Averages the monthly probabilities for each consumer and evaluates
    once per consumer.
    """
    # step-1: monthly probabilities
    probs = predict_consumer_month_probs(
        model, df,
        days_per_example=days_per_example,
        positive_class=positive_class,
        dtype=dtype,
    )

    # step-2: collapse the months -> one p̂ per consumer
    if isinstance(agg, str):
        agg_probs = probs.groupby("consumer_id").agg(agg)
    else:                                   # callable
        agg_probs = probs.groupby("consumer_id").apply(agg)

    y_true = labels.reindex(agg_probs.index).to_numpy()

    if metrics is None:
        metrics = _default_metrics()

    return {name: fn(y_true, agg_probs.to_numpy()) for name, fn in metrics.items()}


from dataclasses import dataclass, field, asdict
from typing import List, Dict, Callable, Any, Optional, Tuple
import numpy as np
import pandas as pd
import pickle


# ← keep your helpers: _default_metrics, optimal_threshold, etc.

@dataclass
class Metrics:
    f1:             float = 0.0
    precision:      float = 0.0
    recall:         float = 0.0
    accuracy:       float = 0.0
    roc_auc:        float = 0.0
    avg_precision:  float = 0.0
    log_loss:       float = 0.0          # optional: expose threshold if useful

    # -----------------------------------------------------------------
    @classmethod
    def from_probs(
        cls,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,                                   # keyword-only
        metrics: Dict[str, Callable] | None = None,
        optimize_on: Tuple[np.ndarray, np.ndarray] | None = None
    ) -> "Metrics":
        """
        Light wrapper around `return_metrics_results`.
        `optimize_on` lets you hand in (y_val, p_val) to pick the best
        threshold on F1 before evaluating `y_true`/`y_prob`.
        """
        scores = return_metrics_results(
            y_true       = y_true,
            y_pred_prob  = y_prob,
            metrics      = metrics,
            optimizeOn   = optimize_on
        )
        # keys from return_metrics_results must match the dataclass fields
        return cls(**scores)

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class SplitResult:
    y_true:       np.ndarray
    y_prob:       np.ndarray
    metric_fns:   Optional[Dict[str, Callable]] = None      # custom metrics
    optimize_on:  Optional[Tuple[np.ndarray, np.ndarray]] = None
    metrics:      Metrics = field(init=False)

    def __post_init__(self):
        self.metrics = Metrics.from_probs(
            self.y_true,
            self.y_prob,
            metrics      = self.metric_fns,
            optimize_on  = self.optimize_on
        )


# ---------------------------------------------------------------------#
# 3.  One fold of training (train / val / test + time)                  
# ---------------------------------------------------------------------#
@dataclass
class FoldResult:
    train:      SplitResult
    val:        SplitResult
    test:       SplitResult
    train_time: float

    # Helpers -----------------------------------------------------------------
    def metrics_df(self) -> pd.DataFrame:
        """Return a tidy dataframe with all metrics for this fold."""
        data = { "set": [], **{k: [] for k in self.train.metrics.as_dict()} }
        for name, split in [("train", self.train), ("val", self.val), ("test", self.test)]:
            data["set"].append(name)
            for k, v in split.metrics.as_dict().items():
                data[k].append(v)
        return pd.DataFrame(data)


# ---------------------------------------------------------------------#
# 4.  All folds for *one* attack type                                   
# ---------------------------------------------------------------------#
@dataclass
class AttackResult:
    attack_type: str
    folds:       List[FoldResult] = field(default_factory=list)

    # -- aggregation shortcuts ----------------------------------------
    def add(self, fold: FoldResult) -> None:
        self.folds.append(fold)

    def mean_metrics(self) -> pd.DataFrame:
        """
        Returns a wide dataframe whose columns are metric_mean / metric_std.
        Works with 1-fold (std = NaN) or N-fold.
        """
        if not self.folds:
            raise ValueError(f"No folds stored for attack {self.attack_type!r}")

        frames = [f.metrics_df().set_index("set") for f in self.folds]
        stacked = pd.concat(frames)                    # (n_folds*3, metrics)

        # mean & std—even with 1 fold, pandas will give std=NaN (which is fine)
        agg = stacked.groupby("set").agg(["mean", "std"])

        # prettier col names: ('f1','mean') → 'f1_mean'
        agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
        return agg.reset_index()                       # 'set' becomes column


# ---------------------------------------------------------------------#
# 5.  Whole experiment: synthetic / real                               
# ---------------------------------------------------------------------#
@dataclass
class ExperimentResults:
    synthetic: Dict[str, AttackResult] = field(default_factory=dict)  # key = attack_type
    real:      Dict[str, AttackResult] = field(default_factory=dict)

    def _store(self, domain: str, attack_type: str, fold_result: FoldResult):
        bucket = getattr(self, domain)
        if attack_type not in bucket:
            bucket[attack_type] = AttackResult(attack_type)
        bucket[attack_type].add(fold_result)

    def add_fold(self,
                 attack_type: str,
                 fold_result_syn: FoldResult,
                 fold_result_real: FoldResult) -> None:
        self._store("synthetic", attack_type, fold_result_syn)
        self._store("real",      attack_type, fold_result_real)

    # Convenience -----------------------------------------------------
    def summary(self, domain: str, attack_type: str) -> pd.DataFrame:
        """
        >>> exp.summary("synthetic", "attack_3")
        """
        return getattr(self, domain)[attack_type].mean_metrics()

    def to_pickle(self, path: str) -> None:
        with open(path, "wb") as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load_pickle(path: str) -> "ExperimentResults":
        with open(path, "rb") as fp:
            return pickle.load(fp)
    def table(self, domain: str, set_name: str = "test") -> pd.DataFrame:
        """
        One row per attack, columns = metric_mean / metric_std for the
        chosen split (train / val / test).  Great for quick comparison.
        """
        records = []
        for atk, atk_res in getattr(self, domain).items():
            df = atk_res.mean_metrics()                 # 3 rows: train/val/test
            row = df[df["set"] == set_name].drop(columns="set")
            row["attack_type"] = atk
            records.append(row)

        if not records:
            raise ValueError(f"No results found for domain={domain!r}")

        return pd.concat(records, ignore_index=True) \
                .set_index("attack_type") \
                .sort_index()
    
    def long_df(self) -> pd.DataFrame:
        """
        Fully exploded dataframe:
        domain | attack | fold | set | metric | value
        """
        rows = []
        for domain in ("synthetic", "real"):
            for attack, atk_res in getattr(self, domain).items():
                for fold_idx, fold in enumerate(atk_res.folds, start=1):
                    for set_name, split in (("train", fold.train),
                                            ("val",   fold.val),
                                            ("test",  fold.test)):
                        for m, v in split.metrics.as_dict().items():
                            rows.append(
                                dict(domain   = domain,
                                    attack   = attack,
                                    fold     = fold_idx,
                                    set      = set_name,
                                    metric   = m,
                                    value    = v)
                            )
        return pd.DataFrame(rows)