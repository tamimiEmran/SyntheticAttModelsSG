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

def optimal_threshold(
    y_true:     np.ndarray,
    y_pred_prob: np.ndarray,
    *,
    metric: str = "f1",
    thresholds: Iterable[float] | None = None,
    return_metrics: bool = False,
) -> float | Tuple[float, Dict[str, float]]:
    
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
