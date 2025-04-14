# src/data/preprocessing.py
"""
Functions for cleaning, transforming, feature engineering, and structuring data
for machine learning models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN
from typing import Tuple, Optional, Dict, Any, Union
from scipy.stats import skew, kurtosis, iqr

# --- Basic Cleaning and Transformation ---

def interpolate_and_fill(df: pd.DataFrame, limit: int = 1, fill_value: float = 0.0) -> pd.DataFrame:
    """
    Applies linear interpolation with a limit and fills remaining NaNs.

    Args:
        df (pd.DataFrame): Input dataframe.
        limit (int): Maximum number of consecutive NaNs to fill via interpolation.
        fill_value (float): Value to fill any remaining NaNs with.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    df = df.interpolate(method='linear', limit=limit, axis=0) # Interpolate along time axis
    df = df.fillna(fill_value)
    return df

def clip_dataframe(df: pd.DataFrame, std_devs: float = 2.0) -> pd.DataFrame:
    """
    Clips values in each column based on mean +/- std_devs * std.

    Args:
        df (pd.DataFrame): Input dataframe (consumers as columns, time as rows).
        std_devs (float): Number of standard deviations for clipping bounds.

    Returns:
        pd.DataFrame: Dataframe with values clipped.
    """
    if df.empty:
        return df
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    upper_bound = mean + std_devs * std
    lower_bound = mean - std_devs * std
    # Clip requires aligning axes correctly if using DataFrame methods
    # Easier to apply per column or transpose if needed
    clipped_df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)
    return clipped_df

def scale_data(
    train_data: np.ndarray,
    test_data: Optional[np.ndarray] = None,
    scaler_type: str = 'standard'
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Any]]:
    """
    Scales the data using StandardScaler or MinMaxScaler. Fits only on train_data.

    Args:
        train_data (np.ndarray): Training data array.
        test_data (Optional[np.ndarray]): Optional testing data array.
        scaler_type (str): Type of scaler ('standard' or 'minmax').

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Any]]:
            - If test_data is None: Scaled training data.
            - If test_data is provided: Scaled training data, scaled testing data, fitted scaler object.
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")

    # Scaler expects 2D array [n_samples, n_features].
    # If data is time series [n_samples, n_timesteps], reshape.
    original_shape_train = train_data.shape
    if train_data.ndim > 1:
         # Flatten features for scaler if needed (e.g., time series)
         # Assuming scaling is applied per feature across samples/time
         # If scaling per time step across samples, adjust reshape
         train_data_reshaped = train_data.reshape(-1, 1) if train_data.ndim == 1 else train_data # Adjust as needed
    else:
         train_data_reshaped = train_data.reshape(-1, 1)


    scaled_train_data_reshaped = scaler.fit_transform(train_data_reshaped)
    scaled_train_data = scaled_train_data_reshaped.reshape(original_shape_train)


    if test_data is not None:
        original_shape_test = test_data.shape
        if test_data.ndim > 1:
            test_data_reshaped = test_data.reshape(-1, 1) if test_data.ndim == 1 else test_data
        else:
            test_data_reshaped = test_data.reshape(-1, 1)

        scaled_test_data_reshaped = scaler.transform(test_data_reshaped)
        scaled_test_data = scaled_test_data_reshaped.reshape(original_shape_test)
        return scaled_train_data, scaled_test_data, scaler
    else:
        return scaled_train_data


# --- Feature Engineering and Example Creation ---

def add_statistical_features(data_array: np.ndarray, axis=1) -> Tuple[np.ndarray, int]:
    """
    Calculates statistical features (mean, std, min, max) and prepends them.

    Args:
        data_array (np.ndarray): Input array (e.g., [n_examples, n_timesteps]).
        axis (int): Axis along which to calculate statistics (default 1 for time series).

    Returns:
        Tuple[np.ndarray, int]:
            - Array with statistical features prepended.
            - Number of features added (currently 4).
    """
    if data_array.size == 0:
        return data_array, 0

    means = np.mean(data_array, axis=axis, keepdims=True)
    stds = np.std(data_array, axis=axis, keepdims=True)
    mins = np.min(data_array, axis=axis, keepdims=True)
    maxs = np.max(data_array, axis=axis, keepdims=True)

    # Handle potential NaNs or empty slices if necessary
    means = np.nan_to_num(means)
    stds = np.nan_to_num(stds)
    # Mins/maxs might raise error on empty slice, check shape or use nanmin/nanmax
    if data_array.shape[axis] == 0:
         mins = np.full_like(means, 0) # Or appropriate default
         maxs = np.full_like(means, 0)


    # Ensure features have the correct shape for hstack if axis=1
    if axis == 1:
        if data_array.ndim == 1: # Single example case
             data_array = data_array.reshape(1, -1)
             means = means.reshape(1, 1)
             stds = stds.reshape(1, 1)
             mins = mins.reshape(1, 1)
             maxs = maxs.reshape(1, 1)
        else: # Multiple examples
             means = means.reshape(-1, 1)
             stds = stds.reshape(-1, 1)
             mins = mins.reshape(-1, 1)
             maxs = maxs.reshape(-1, 1)

        # Add more stats if needed
        # skewness = np.apply_along_axis(skew, axis, data_array).reshape(-1, 1)
        # kurt = np.apply_along_axis(kurtosis, axis, data_array).reshape(-1, 1)
        # median = np.median(data_array, axis=axis).reshape(-1, 1)
        # iqr_val = np.apply_along_axis(iqr, axis, data_array).reshape(-1, 1)

        # features = np.hstack((means, stds, mins, maxs, skewness, kurt, median, iqr_val))
        features = np.hstack((means, stds, mins, maxs))
        num_features_added = features.shape[1]
        featured_array = np.hstack((features, data_array))


    else:
        # Handle axis=0 if needed, adjusting reshape/hstack accordingly
        raise NotImplementedError("Statistical features for axis=0 not implemented")


    return featured_array, num_features_added


def pad_array(data_array: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pads the array along the last dimension to reach the target_length.
    Padding is added to the end (right side).

    Args:
        data_array (np.ndarray): Input array (e.g., [n_examples, n_current_length]).
        target_length (int): The desired length after padding.

    Returns:
        np.ndarray: Padded array.
    """
    if data_array.size == 0:
        return np.zeros((data_array.shape[0], target_length)) # Return zeros if empty input

    current_length = data_array.shape[-1]
    padding_needed = max(0, target_length - current_length)

    if padding_needed == 0:
        return data_array

    # Define padding widths: ((before_axis1, after_axis1), (before_axis2, after_axis2), ...)
    # We only pad the last axis, after the existing data.
    pad_widths = [(0, 0)] * (data_array.ndim - 1) + [(0, padding_needed)]

    padded_array = np.pad(data_array, pad_width=pad_widths, mode='constant', constant_values=0.0)
    return padded_array

def create_monthly_examples(
    df: pd.DataFrame,
    labels_series: pd.Series,
    add_stats: bool = True,
    pad_length: int = 31 # Default based on longest month
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates monthly examples from a DataFrame.
    Assumes df has datetime index and consumers as columns.
    labels_series is indexed by consumer ID.

    Args:
        df (pd.DataFrame): Input DataFrame with datetime index, consumers as columns.
        labels_series (pd.Series): Series mapping consumer ID (column names) to labels.
        add_stats (bool): Whether to prepend statistical features.
        pad_length (int): Target length for each monthly example after padding.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Numpy array of examples and corresponding labels.
    """
    if df.empty or labels_series.empty:
        return np.array([]), np.array([])

    df = df.copy()
    df['Year-Month'] = df.index.to_period('M')
    grouped = df.groupby('Year-Month')

    all_examples = []
    all_labels = []
    num_features_added = 0

    consumer_columns = df.columns.drop('Year-Month') # Get consumer IDs

    for group_name, group_data in grouped:
        # Process only consumer columns for this month
        month_consumer_data = group_data[consumer_columns]

        # Transpose: shape becomes [n_consumers, n_days_in_month]
        example_month = month_consumer_data.values.T

        # Get labels for consumers present in this month's data
        # Match labels using the consumer column index
        consumer_ids_in_group = month_consumer_data.columns
        labels_month = labels_series.reindex(consumer_ids_in_group).values

        # Handle cases where a consumer might be missing a label (though should not happen with reindex)
        if np.isnan(labels_month).any():
             print(f"Warning: Missing labels for some consumers in month {group_name}. Filling with -1.")
             labels_month = np.nan_to_num(labels_month, nan=-1) # Or skip these consumers

        if example_month.size > 0:
            # Add statistical features if requested
            if add_stats:
                example_month, num_features_added = add_statistical_features(example_month, axis=1)

            # Pad to ensure uniform length
            target_feature_length = pad_length + num_features_added
            example_month_padded = pad_array(example_month, target_feature_length)

            all_examples.append(example_month_padded)
            all_labels.append(labels_month)

    if not all_examples:
        return np.array([]), np.array([])

    # Combine examples and labels from all months
    examples_np = np.vstack(all_examples)
    labels_np = np.concatenate(all_labels)

    return examples_np, labels_np.astype(int) # Ensure labels are integers

def create_daily_examples(
    df: pd.DataFrame,
    labels_series: pd.Series,
    readings_per_day: int = 48,
    add_stats: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates daily examples from a DataFrame.
    Assumes df has datetime index and consumers as columns.
    labels_series is indexed by consumer ID.

    Args:
        df (pd.DataFrame): Input DataFrame with datetime index, consumers as columns.
        labels_series (pd.Series): Series mapping consumer ID to labels.
        readings_per_day (int): Number of readings expected per day (e.g., 48 for 30-min intervals).
        add_stats (bool): Whether to prepend statistical features to each daily example.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Numpy array of daily examples and corresponding labels.
    """
    if df.empty or labels_series.empty:
        return np.array([]), np.array([])

    # Transpose to have consumers as rows, time as columns
    df_transposed = df.T

    # Check if number of readings is multiple of readings_per_day
    total_readings = df_transposed.shape[1]
    if total_readings % readings_per_day != 0:
        print(f"Warning: Total readings ({total_readings}) not divisible by readings_per_day ({readings_per_day}). Reshaping might be incorrect.")
        # Optionally, trim or pad df_transposed.shape[1] here

    num_consumers = df_transposed.shape[0]
    num_days = total_readings // readings_per_day

    # Reshape into [num_consumers * num_days, readings_per_day]
    daily_examples_raw = df_transposed.values.reshape(num_consumers * num_days, readings_per_day)

    # Create corresponding labels
    consumer_ids = df_transposed.index
    labels_array = labels_series.reindex(consumer_ids).values
    daily_labels = np.repeat(labels_array, num_days)

    # Add statistical features if requested
    if add_stats:
        daily_examples_featured, _ = add_statistical_features(daily_examples_raw, axis=1)
    else:
        daily_examples_featured = daily_examples_raw

    # Shuffle examples and labels together
    permutation = np.random.permutation(len(daily_labels))
    shuffled_examples = daily_examples_featured[permutation]
    shuffled_labels = daily_labels[permutation]

    return shuffled_examples, shuffled_labels.astype(int)


# --- Filtering and Sampling ---

def filter_excessive_zeros(
    examples: np.ndarray,
    labels: np.ndarray,
    max_zeros: int,
    tolerance: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters out examples (rows) that contain more than max_zeros zero readings.

    Args:
        examples (np.ndarray): Input examples array [n_examples, n_features].
        labels (np.ndarray): Corresponding labels array [n_examples].
        max_zeros (int): Maximum allowed number of zero readings per example.
        tolerance (float): Tolerance for checking closeness to zero.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered examples and labels.
    """
    if examples.size == 0:
        return examples, labels

    # Count zeros (or values close to zero) along the feature axis (axis=1)
    zero_counts = np.sum(np.isclose(examples, 0.0, atol=tolerance), axis=1)
    valid_indices = zero_counts <= max_zeros

    return examples[valid_indices], labels[valid_indices]

def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Applies SMOTE oversampling."""
    smote = SMOTE(random_state=random_state, **kwargs)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def apply_adasyn(X: np.ndarray, y: np.ndarray, random_state: int = 42, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Applies ADASYN oversampling."""
    adasyn = ADASYN(random_state=random_state, **kwargs)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled

# --- Specific Preprocessing Techniques (from main_evaluate.py) ---

def preprocess_tsg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies preprocessing steps similar to TSG paper (interpolation, fillna(0),
    clipping, MinMaxScaler).
    Assumes df has consumers as columns, time as rows.
    """
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    df_processed = interpolate_and_fill(df_numeric, limit=1, fill_value=0.0)

    # Clipping requires mean/std calculated correctly (usually per consumer)
    # Transpose, calculate stats, clip, transpose back might be needed if original df is time-indexed
    # If df is already consumers-as-columns:
    df_processed = clip_dataframe(df_processed, std_devs=2.0)

    # Scale data using MinMaxScaler
    scaler = MinMaxScaler()
    # Scaler expects samples as rows, features as columns.
    # If df has time as rows, consumers as columns, scale each consumer's series
    scaled_data = scaler.fit_transform(df_processed)
    df_scaled = pd.DataFrame(scaled_data, index=df_processed.index, columns=df_processed.columns)

    if df_scaled.isna().sum().sum() != 0:
        print('Warning: NaNs detected after TSG preprocessing and scaling.')

    return df_scaled

def _calculate_energy(temp_values: np.ndarray, mean: float) -> float:
    """Helper for preprocess_energies."""
    mask = temp_values >= mean
    # Original code used 0.1 factor, unclear origin, keeping for consistency
    return np.sum(temp_values[mask] * 0.1)

def preprocess_energies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the 'energies' preprocessing technique from main_evaluate.py.
    Uses a rolling window approach to update values based on local means and energy calculation.
    Assumes df has consumers as columns, time as rows.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    df_processed = df.copy()

    for col in df.columns:
        series = df_processed[col].copy()
        values = series.values
        gmean = np.nanmean(values) # Global mean for fallback

        def _update(ix):
            # Define window, handle boundaries
            start = max(0, ix - 5)
            end = min(len(values), ix + 5) # Ensure index is within bounds for slicing
            temp_values = values[start:end]

            # Calculate mean, handle NaNs
            local_mean = np.nanmean(temp_values)
            if isnan(local_mean):
                 local_mean = gmean # Use global mean if local window is all NaN
                 if isnan(local_mean): # Handle case where entire series might be NaN
                      return 0 # Or some other default
                 temp_values_filled = np.full_like(temp_values, local_mean) # Fill temp window for calculation
                 factor = _calculate_energy(temp_values_filled, local_mean)
            else:
                 factor = _calculate_energy(temp_values, local_mean)


            return factor * local_mean

        # Iterate and update values - Original code updated only at specific indices
        # This implementation updates potentially NaN values within windows
        # Consider if the original intent was different (e.g., only update specific NaN indices)
        # Original code iterated range(5, len(index) - 4, 10) and updated series.iloc[ix] if isnan
        # This suggests a sparser update. Replicating that precisely:
        indices_to_update = range(5, len(series.index) - 4, 10)
        for ix_pos in indices_to_update:
             if pd.isna(series.iloc[ix_pos]):
                  series.iloc[ix_pos] = _update(ix_pos)


        # Forward fill remaining NaNs after targeted updates
        series.ffill(inplace=True)
        # Backward fill any NaNs at the beginning
        series.bfill(inplace=True)
        df_processed[col] = series

    return df_processed
