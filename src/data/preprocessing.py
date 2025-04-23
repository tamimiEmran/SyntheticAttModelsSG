# there are three main preprocessing steps:
# 1. be mindful of folds (train, val, test)
# 2. preprocess the dataset (e.g., fill NaNs, scale, add stats)
# 3. creates examples from the dataframe.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple,List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
def filter_examples_with_excessive_zeros(examples, max_zeros, tolerance=1e-5):
    zero_counts = np.sum(np.isclose(examples, 0.0, atol=tolerance), axis=1)
    return examples[zero_counts <= max_zeros]
def preprocess_tsg(df: pd.DataFrame,
                   cap_sigma: float = 3.0) -> pd.DataFrame:
    """
    SGCC-style preprocessing for a *single* wide daily-load DataFrame.

    Steps applied **independently to every consumer (row)**:

    1. Ensure numeric, then interpolate single missing values (per row),
       filling any remaining NaNs with 0.
    2. Cap extreme spikes at  (mean + cap_sigma · std)  of that row.
    3. Min-Max scale each row to the range [0, 1].

    Parameters
    ----------
    df : pd.DataFrame
        Shape (n_consumers, n_days).  Index = consumer IDs,
        columns = daily `DatetimeIndex` (or strings that can be parsed).
    cap_sigma : float, default 3.0
        Multiplier for the row-wise standard deviation when computing
        the clipping threshold.

    Returns
    -------
    pd.DataFrame
        Pre-processed DataFrame, same shape & labels as the input.
    """
    

    # 1) copy & force numeric
    out = df.copy().apply(pd.to_numeric)

    # interpolate *along the day axis* (axis=1) – limit 1 consecutive NaN
    out.interpolate(axis=1, limit=1, inplace=True)
    out.fillna(0, inplace=True)

    # 2) row-wise clipping
    means = out.mean(axis=1)
    stds  = out.std(axis=1)
    thresholds = means + cap_sigma * stds
    # clip upper bound; align on index (axis=0)
    out = out.clip(upper=thresholds, axis=0)

    # 3) row-wise min-max scaling
    mins = out.min(axis=1)
    ranges = (out.max(axis=1) - mins).replace(0, 1)   # avoid division by zero
    out = out.sub(mins, axis=0).div(ranges, axis=0)

    # sanity-check
    if out.isna().any().any():
        raise ValueError("NaNs remain after preprocessing.")

    return out

def pad_and_stats(arr_list: List[np.ndarray],
                  days_per_example: int = 31) -> np.ndarray:
    """
    FOR SGCC dataset:
    Given a list of 1D arrays (each shape: (n_timepoints,)):
      1. Compute per-array stats: mean, std, min, max
      2. Prepend those 4 stats to the array
      3. Pad with zeros on the right to reach (4 + days_per_example) length
    Returns a 2D array of shape (n_examples, 4 + days_per_example).
    """
    if not isinstance(arr_list, (list, tuple)):
        raise ValueError("Input must be a list (or tuple) of 1D numpy arrays.")

    stats_count = 4
    target_width = stats_count + days_per_example
    processed = []

    for i, arr in enumerate(arr_list):
        arr = np.asarray(arr)
        if arr.ndim != 1:
            raise ValueError(f"Element {i} is not 1D (got ndim={arr.ndim}).")

        # 1) compute stats
        m = arr.mean()
        s = arr.std()
        mn = arr.min()
        mx = arr.max()

        # 2) prepend stats + original data
        with_stats = np.hstack(([m, s, mn, mx], arr))

        # 3) pad (or error if too long)
        pad_width = target_width - with_stats.size
        if pad_width < 0:
            raise ValueError(
                f"Example {i}: length {with_stats.size} > target {target_width}. "
                "Either truncate your inputs or raise days_per_example."
            )

        padded = np.pad(with_stats,
                        pad_width=(0, pad_width),
                        mode='constant',
                        constant_values=0.0)

        processed.append(padded)

    # stack into (n_examples, 4 + days_per_example)
    return np.vstack(processed)
def create_monthly_examples(
    df: pd.DataFrame,
    labels: pd.Series,
    days_per_example: int = 31
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice a wide daily-timeseries DataFrame into consumer-month examples,
    compute stats + padding via `pad_and_stats`, and collect labels.

    Parameters
    ----------
    df : pd.DataFrame
        shape = (n_consumers, n_days_total)
        index = consumer IDs, columns = a DatetimeIndex of daily dates
    labels : pd.Series
        consumer-level labels, indexed by the same consumer IDs as df.index
    days_per_example : int
        the target number of days to pad each month up to (default=31)

    Returns
    -------
    X : np.ndarray, shape = (n_consumers * n_months, 4 + days_per_example)
        each row = [mean, std, min, max, day₁, …, day_K, 0…] for one consumer-month
    y : np.ndarray, shape = (n_consumers * n_months,)
        the consumer label repeated for each month
    """
    # map each column to its month period
    col_periods = df.columns.to_period("M")
    all_periods = col_periods.unique()

    X_list = []
    y_list = []

    for period in all_periods:
        # select the columns for this month
        mask = (col_periods == period)
        month_cols = df.columns[mask]
        if len(month_cols) == 0:
            continue

        # subset to this month
        month_df = df[month_cols]

        # build list of 1D arrays: one per consumer
        arr_list = list(month_df.to_numpy())  # each is shape (n_days_in_month,)

        # stats-augment + pad each row
        X_month = pad_and_stats(arr_list, days_per_example)

        # grab each consumer's label
        # (reindex in case some IDs are missing)
        y_month = labels.reindex(month_df.index).to_numpy()

        X_list.append(X_month)
        y_list.append(y_month)

    # stack all months × consumers
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y
def oversample(x, y):
    ada = ADASYN(sampling_strategy='minority', random_state=42)
    X_resampled, y_resampled = ada.fit_resample(x, y)
    return X_resampled, y_resampled

