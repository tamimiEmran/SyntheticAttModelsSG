import numpy as np
import pandas as pd


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
    # ----- 1.  Group columns by month -------------------------------------
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

    # ----- 3.  Wrap up in a tidy Series -----------------------------------
    idx = pd.MultiIndex.from_tuples(index_parts, names=["consumer_id", "period"])
    return pd.Series(probs, index=idx, name="probability").sort_index()


from typing import Callable, Dict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, log_loss
)

# ------------------------------------------------------------------
# sensible defaults (feel free to swap / extend)
# ------------------------------------------------------------------
def _default_metrics() -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """
    Returns a dictionary {metric_name: callable} where each callable has the
    signature f(y_true, y_pred_prob) and returns a float.
    """
    return dict(
        roc_auc        = roc_auc_score,
        avg_precision  = average_precision_score,
        accuracy       = lambda y, p: accuracy_score(y, p >= 0.5),
        log_loss       = lambda y, p: log_loss(
            y, np.c_[1 - p, p], labels=[0, 1]
        ),
    )

# ------------------------------------------------------------------
# 1.  Evaluate directly on consumer-month predictions
# ------------------------------------------------------------------
def evaluate_monthly(
    model,
    df: pd.DataFrame,
    labels: pd.Series,
    *,
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

# ------------------------------------------------------------------
# 2.  Aggregate → one probability per consumer, then evaluate
# ------------------------------------------------------------------
def evaluate_aggregated(
    model,
    df: pd.DataFrame,
    labels: pd.Series,
    *,
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
