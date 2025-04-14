# src/attack_models/implementations.py
"""
Concrete implementations of various synthetic attack models.

Each class inherits from BaseAttackModel and implements a specific attack type.
The logic is now adapted from the provided original `attackTypes.py`.

*** NOTE: Ensure ATTACK_CONSTANTS in experiments/config.py are correctly set
    from the original init.py file for accurate replication. ***

*** WARNING: Attack Type 12's monthly implementation retains significant complexity
    and relies on weekly/daily patterns within the monthly data. Its behavior might
    still subtly differ if the exact grouping/indexing isn't perfectly replicated. ***
"""

import pandas as pd
import numpy as np
import random
from typing import Union

from .base import BaseAttackModel
from experiments.config import ATTACK_CONSTANTS # Import the constants
def week_in_month_year(dates: pd.Series) -> pd.Series:
    """
    Get a unique identifier for weeks that respects year and month boundaries.
    Returns a string identifier in format 'YYYY-MM-W' where W is the week number (1-based) within that month.
    
    Args:
        dates: DatetimeIndex or Series of datetime values
    
    Returns:
        Series of strings identifying each unique week in format 'YYYY-MM-W'
    """
    # Ensure dates are datetime objects
    if isinstance(dates, pd.DatetimeIndex):
        dates = pd.Series(dates)  # Convert DatetimeIndex to Series for .dt accessor
    elif not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates, errors='coerce')
    
    # Check for NaN values
    if dates.isna().any():
        raise ValueError("Could not convert dates index to datetime in week_in_month_year")

    # Calculate the first day of the month for each date
    firstday_in_month = dates - pd.to_timedelta(dates.dt.day - 1, unit='d')
    
    # Calculate the week number within the month (1-based)
    week_num = (dates.dt.day - 1 + firstday_in_month.dt.weekday) // 7 + 1
    
    # Create unique identifier combining year, month, and week number
    year_month_week = (dates.dt.year.astype(str) + '-' + 
                       dates.dt.month.astype(str).str.zfill(2) + '-W' + 
                       week_num.astype(str))
    
    return year_month_week


class AttackType0(BaseAttackModel):
    """Attack Type 0: Zero consumption (Original _attack0)"""
    attack_id = "0"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        return series * 0

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sets all consumption values to zero."""
        # .apply applies the function column-wise (axis=0)
        return data.apply(self._core_attack_logic, axis=0)

class AttackType1(BaseAttackModel):
    """Attack Type 1: Scale consumption by a random factor (Original _attack1)"""
    attack_id = "1"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        rand_scale = ATTACK_CONSTANTS["attack1range"]
        return series * random.uniform(rand_scale[0], rand_scale[1])

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Multiplies consumption values by a random factor."""
        return data.apply(self._core_attack_logic, axis=0)

class AttackType2(BaseAttackModel):
    """Attack Type 2: Set a random window to zero (Original _attack2)"""
    attack_id = "2"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        series_copy = series.copy() # Work on a copy to use .update safely
        window_size = ATTACK_CONSTANTS["AttackReading"]
        if len(series_copy.index) <= window_size:
            # If series is too short, zero out everything or return as is?
            # Original might error or behave unpredictably. Zeroing seems safer.
            return series_copy * 0
        try:
             start = random.randint(0, len(series_copy.index) - window_size - 1)
        except ValueError: # Handle case where window_size = len - 1
             start = 0
        end = start + window_size
        # Create a series of zeros with the correct index slice
        zero_window = pd.Series(0.0, index=series_copy.index[start:end])
        series_copy.update(zero_window) # Update the copy
        return series_copy

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sets a contiguous random window of consumption values to zero."""
        return data.apply(self._core_attack_logic, axis=0)

class AttackType3(BaseAttackModel):
    """Attack Type 3: Apply per-cell random scaling (Original _attack3)"""
    attack_id = "3"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        rand_scale = ATTACK_CONSTANTS["attack1range"]
        # Apply scaling factor to each element individually
        return series.apply(lambda cell: cell * random.uniform(rand_scale[0], rand_scale[1]))

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies a *different* random scaling factor to each individual reading."""
        return data.apply(self._core_attack_logic, axis=0)

class AttackType4(BaseAttackModel):
    """Attack Type 4: Scale a random window (Original _attack4)"""
    attack_id = "4"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        series_copy = series.copy()
        window_size = ATTACK_CONSTANTS["AttackReading"]
        attack_range = ATTACK_CONSTANTS["attack4"]
        if len(series_copy.index) <= window_size:
            # Apply factor to whole series if window doesn't fit? Or return original?
            # Let's apply to whole series for simplicity, mimicking potential edge case.
             scaling_factor = random.uniform(attack_range[0], attack_range[1])
             return series_copy * scaling_factor
        try:
             start = random.randint(0, len(series_copy.index) - window_size - 1)
        except ValueError:
             start = 0
        end = start + window_size
        scaling_factor = random.uniform(attack_range[0], attack_range[1])
        # Create scaled window and update
        scaled_window = series_copy.iloc[start:end] * scaling_factor
        series_copy.update(scaled_window)
        return series_copy

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scales a contiguous random window of consumption by a random factor."""
        return data.apply(self._core_attack_logic, axis=0)

class AttackType5(BaseAttackModel):
    """Attack Type 5: Set values based on mean and random factor (Original _attack5)"""
    attack_id = "5"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        attack_range = ATTACK_CONSTANTS["attack5"]
        series_mean = series.mean()
        if pd.isna(series_mean): series_mean = 0 # Handle potential NaN mean
        # Set each cell based on the series mean * random factor
        return series.apply(lambda cell: series_mean * random.uniform(attack_range[0], attack_range[1]))

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sets each reading to [series_mean * random_factor] independently."""
        return data.apply(self._core_attack_logic, axis=0)

class AttackType6(BaseAttackModel):
    """Attack Type 6: Clip high values to a threshold (Original _attack6)"""
    attack_id = "6"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        series_copy = series.copy()
        rand_scale = ATTACK_CONSTANTS["attack1range"]
        series_max = series.max()
        if pd.isna(series_max) or series_max <= 0: # Handle non-positive max
             return series_copy # Or return zeros? Return copy for now.
        threshold = series_max * random.uniform(rand_scale[0], rand_scale[1])
        # Find values above threshold
        attack_this_mask = series_copy > threshold
        # Set these values TO the threshold
        series_copy[attack_this_mask] = threshold
        return series_copy

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clips values exceeding a random threshold (based on max) to that threshold."""
        return data.apply(self._core_attack_logic, axis=0)

class AttackType7(BaseAttackModel):
    """Attack Type 7: Subtract threshold, floor at zero (Original _attack7)"""
    attack_id = "7"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        rand_scale = ATTACK_CONSTANTS["attack1range"]
        series_max = series.max()
        if pd.isna(series_max) or series_max <= 0: # Handle non-positive max
             return series.copy() * 0 # Subtracting from 0 or less makes little sense, return 0s?
        threshold = series_max * random.uniform(rand_scale[0], rand_scale[1])
        series_subtracted = series - threshold
        series_subtracted[series_subtracted < 0.0] = 0.0 # Floor at zero
        return series_subtracted

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Subtracts a random threshold (based on max) and sets negative results to 0."""
        return data.apply(self._core_attack_logic, axis=0)

class AttackType8(BaseAttackModel):
    """Attack Type 8: Ramp-down attack within a window (Original _attack8)"""
    attack_id = "8"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        series_copy = series.copy() # Work on a copy
        window_size = ATTACK_CONSTANTS["AttackReading"]
        i_max_range = ATTACK_CONSTANTS["i_max"]
        roc_range = ATTACK_CONSTANTS["rocAttIntense"]

        if len(series_copy.index) <= window_size:
             return series_copy # Cannot apply ramp if too short

        try:
             start = random.randint(0, len(series_copy.index) - window_size - 1)
        except ValueError:
             start = 0
        end = start + window_size
        i_max = random.uniform(i_max_range[0], i_max_range[1])

        # Original ROC calculation: rocAttIntense * (i_max / window_size)
        # Recalculating rocAttIntense based on this seems unusual. Let's assume
        # rocAttIntense is the base rate of change per step within the window.
        roc_intense_per_step = random.uniform(roc_range[0], roc_range[1])

        # Apply the logic element-wise using index position
        original_values = series_copy.values
        modified_values = original_values.copy()

        for idx_pos in range(len(modified_values)):
            value = original_values[idx_pos]
            if idx_pos >= start and idx_pos < end: # Ramp down within window
                # Factor decreases linearly from 1 towards (1 - factor_at_end)
                # Let's try to match the effect: total reduction reaches ~i_max at the end
                # Reduction at step 'k' = k * rate. Max reduction = window_size * rate = i_max
                rate = i_max / window_size # Effective rate to reach i_max reduction
                reduction_factor = (idx_pos - start + 1) * rate # +1 because original iterated to <= end
                modified_values[idx_pos] = value * max(0, 1 - reduction_factor) # Ensure factor >= 0
            elif idx_pos >= end: # Apply max reduction after window
                modified_values[idx_pos] = value * (1 - i_max)
            # else: pass (before start window)

        return pd.Series(modified_values, index=series_copy.index)


    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies a ramp-down modification within a random window."""
        return data.apply(self._core_attack_logic, axis=0)

class AttackType9(BaseAttackModel):
    """Attack Type 9: Set to mean (Original _attack9)"""
    attack_id = "9"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        series_mean = series.mean()
        if pd.isna(series_mean): series_mean = 0
        # Create a new series filled with the mean
        s_updated = pd.Series(series_mean, index=series.index)
        return s_updated

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sets each consumer's consumption to their mean value for the period."""
        return data.apply(self._core_attack_logic, axis=0)

class AttackType10(BaseAttackModel):
    """Attack Type 10: Reverse time order within each week (Original _attack10Month)"""
    attack_id = "10"
    # Note: Original _attack10 reversed the whole series.
    # The changeMonth function used _attack10Month logic for type 10.

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reverses the time order of consumption values within each week of the month."""
        if not isinstance(data.index, pd.DatetimeIndex):
             print("Warning: Attack Type 10 requires a DatetimeIndex. Attempting conversion.")
             try:
                  data.index = pd.to_datetime(data.index)
             except Exception as e:
                  print(f"Error converting index to DatetimeIndex for Attack 10: {e}. Returning original data.")
                  raise ValueError("Index conversion failed.")

        # Get readings for the first day in the dataset
               

        day_data = data.index.date[0]
        len_day_data = len(data.index[data.index.date == day_data])
        modified_data = data.copy()
        # If readings per day > 32, it's likely Ausgrid (30-min intervals)
        # Otherwise, it's likely SGCC (daily readings)
        if len_day_data > 1:
            print(f"ausgrid: data head {modified_data.head()}")
            weekly_groups = modified_data.groupby(modified_data.index.date)
            print(f"ausgrid: The number of grouped data is {len(weekly_groups)}")
            print(f"ausgrid: groups mean is {weekly_groups.mean().head()}")
        else:
            print(f"SGCC: data head {modified_data.head()}")
            weekly_groups = modified_data.groupby(week_in_month_year(modified_data.index))
            print(f"SGCC: The number of grouped data is {len(weekly_groups)}")
            print(f"SGCC: groups mean is {weekly_groups.mean().head()}")



        
        # Group by week number within the month
        # Note: weekinmonth function assumes the index is a DatetimeIndex and that the dataset is sgcc


        processed_weeks = []
        original_indices = []
        for week_num, week_data in weekly_groups:
             if not week_data.empty:
                  
                  flipped_values = week_data.values[::-1] # Reverse the values for this week
                  # Create a new DataFrame with flipped values but original index/columns for this week
                  flipped_week = pd.DataFrame(flipped_values, index=week_data.index, columns=week_data.columns)
                  processed_weeks.append(flipped_week)
                  original_indices.extend(week_data.index.tolist()) # Keep track of indices

        if not processed_weeks:
             return data # Return original if no weeks processed

        # Concatenate processed weeks and sort by original index to restore order
        result_df = pd.concat(processed_weeks)
        # Reindex based on the collected original indices to ensure perfect alignment
        result_df = result_df.reindex(data.index)

        # Handle potential NaNs introduced by reindexing if some weeks were empty/skipped
        result_df = result_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        return result_df


class AttackType11(BaseAttackModel):
    """Attack Type 11: Redistribute energy from peak window (Original _attack11)"""
    attack_id = "11"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        series_copy = series.copy()
        time_index = series_copy.index # Store original index
        series_copy.index = range(len(series_copy)) # Use integer index for rolling/idxmax

        interval = ATTACK_CONSTANTS["AttackReading"]
        reduction_factor = ATTACK_CONSTANTS["attack11Factor"]

        if len(series_copy) < interval:
            # Not enough data for the window, return original or scaled?
            # Original might error. Let's return original.
            series_copy.index = time_index
            return series_copy

        # Find the window with the maximum sum
        rolling_sum = series_copy.rolling(window=interval).sum()
        if rolling_sum.isna().all(): # Handle case where series might be all NaN?
            series_copy.index = time_index
            return series_copy
        end_pos = rolling_sum.idxmax() # Position index (0 to N-1)
        start_pos = max(0, end_pos - interval + 1) # Ensure start isn't negative

        # Get the actual index values corresponding to start/end positions
        start_idx = time_index[start_pos]
        end_idx = time_index[end_pos]

        # Select the period based on original time_index
        period_mask = (series_copy.index >= start_pos) & (series_copy.index <= end_pos)
        non_attack_mask = ~period_mask

        periodOfAttack = series_copy[period_mask]
        totalEnergyStolen = periodOfAttack.sum() * (1 - reduction_factor) # Energy removed
        periodOfAttack_new = periodOfAttack * reduction_factor # Scale down the attack period

        series_copy.loc[period_mask] = periodOfAttack_new

        periodOfNonAttack = series_copy[non_attack_mask]
        num_non_attack_points = len(periodOfNonAttack)

        if num_non_attack_points > 0 and totalEnergyStolen > 0:
            energy_to_add_per_point = totalEnergyStolen / num_non_attack_points
            periodOfNonAttack_new = periodOfNonAttack + energy_to_add_per_point
            series_copy.loc[non_attack_mask] = periodOfNonAttack_new

        series_copy.index = time_index # Restore original index
        return series_copy

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Finds peak energy window, reduces it, adds stolen energy elsewhere."""
        return data.apply(self._core_attack_logic, axis=0)


class AttackType12(BaseAttackModel):
    """
    Attack Type 12: Swap consumption profiles based on sums within weeks (Original _attack12Month).

    *** WARNING: This implementation attempts to replicate the complex logic of
    the original `_attack12Month` and `_attack12`. It involves grouping by week,
    calculating daily sums within each week for all consumers, finding a target
    consumer with a lower sum for each original consumer, and swapping the daily
    profile for that week. This logic is complex and sensitive to the exact daily
    structure and consumer pool within the input `data` (assumed monthly).
    Its behavior might differ from the original if the context or data differs.
    Use with caution and verify its effects carefully. ***
    """
    attack_id = "12"

    def _core_attack12_daily_swap(self, daily_df_week: pd.DataFrame) -> pd.DataFrame:
        """
        Performs the core daily profile swapping logic for a given week's data.
        Args:
            daily_df_week (pd.DataFrame): DataFrame for one week (Index=Time, Columns=Consumers).
        Returns:
            pd.DataFrame: DataFrame with profiles swapped for that week.
        """
        if daily_df_week.empty or daily_df_week.shape[1] <= 1:
             return daily_df_week # Cannot swap if empty or only one consumer

        # Calculate daily sums for the week and sort consumers by sum
        daily_sums = daily_df_week.sum(axis=0).sort_values() # Series: Index=Consumer, Value=Sum
        factor = ATTACK_CONSTANTS["attack12Factor"]
        attacked_df_week = daily_df_week.copy() # Create copy to modify

        # For each consumer column in the original week data
        for consumer_col in daily_df_week.columns:
            current_sum = daily_sums.get(consumer_col, np.inf) # Get sum for this consumer

            # Find potential swap targets: consumers with sum < current_sum / factor
            # Exclude the consumer itself from potential targets
            potential_targets = daily_sums[daily_sums < current_sum / factor].drop(consumer_col, errors='ignore')

            if not potential_targets.empty:
                # Select the target with the highest sum among the valid candidates
                # (Original code picked index[-1] of filtered sorted list)
                swap_target_consumer = potential_targets.index[-1]
                # Replace the current consumer's data with the target's data FOR THIS WEEK
                attacked_df_week[consumer_col] = daily_df_week[swap_target_consumer].values
            # else: No suitable swap target found, keep original data for this consumer/week

        return attacked_df_week


    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies the weekly profile swapping logic across the input month data."""
        if not isinstance(data.index, pd.DatetimeIndex):
             print("Warning: Attack Type 12 requires a DatetimeIndex. Attempting conversion.")
             try:
                  data.index = pd.to_datetime(data.index)
             except Exception as e:
                  print(f"Error converting index to DatetimeIndex for Attack 12: {e}. Returning original data.")
                  return data
             
        modified_data = data.copy()
        # Get readings for the first day in the dataset
        day_data = data.index.day[0]
        len_day_data = len(data.index[data.index.day == day_data])

        # If readings per day > 32, it's likely Ausgrid (30-min intervals)
        # Otherwise, it's likely SGCC (daily readings)
        if len_day_data > 32:
            weekly_groups = modified_data.groupby(modified_data.index.date)
        else:
            weekly_groups = modified_data.groupby(week_in_month_year(modified_data.index))




        
        # Group by week number within the month
        

        processed_weeks = []
        for week_num, week_data in weekly_groups:
             if not week_data.empty:
                  # Apply the core daily swapping logic to this week's data
                  processed_week_data = self._core_attack12_daily_swap(week_data)
                  processed_weeks.append(processed_week_data)
             # else: Keep empty weeks as they are (will be handled by concat/reindex)

        if not processed_weeks:
             return data # Return original if no weeks were processed

        # Concatenate processed weeks
        result_df = pd.concat(processed_weeks)
        # Reindex to match the original monthly DataFrame structure and fill gaps
        result_df = result_df.reindex(data.index)
        result_df = result_df.fillna(method='ffill').fillna(method='bfill').fillna(0) # Fill any NaNs

        return result_df


class AttackTypeIEEE(BaseAttackModel):
    """
    Attack Type IEEE: Randomly zero out individual readings (Original _attackIEEE).

    Multiplies each reading by either 0 or 1 randomly.
    """
    attack_id = "ieee"

    def _core_attack_logic(self, series: pd.Series) -> pd.Series:
        # Generate random bits (0 or 1) for each element
        random_mask = np.random.randint(0, 2, size=series.shape, dtype=bool)
        # Multiply series by the boolean mask (True=1, False=0)
        return series * random_mask

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies random zeroing to individual readings for each consumer."""
        return data.apply(self._core_attack_logic, axis=0)


# --- Registry ---
# List of all implemented attack classes
_ALL_ATTACK_CLASSES = [
    AttackType0, AttackType1, AttackType2, AttackType3, AttackType4,
    AttackType5, AttackType6, AttackType7, AttackType8, AttackType9,
    AttackType10, AttackType11, AttackType12, AttackTypeIEEE
]

# --- Self-Correction/Verification ---
# Check if all classes have unique string IDs
all_ids = [cls().attack_id for cls in _ALL_ATTACK_CLASSES]
if len(all_ids) != len(set(all_ids)):
    # This should not happen with the current list unless IDs are duplicated
    import collections
    duplicates = [item for item, count in collections.Counter(all_ids).items() if count > 1]
    raise RuntimeError(f"Duplicate attack_id found in implementations: {duplicates}")
# Ensure all IDs are strings
if not all(isinstance(id_val, str) for id_val in all_ids):
     raise RuntimeError(f"Not all attack_ids are strings: {all_ids}")

print(f"Successfully loaded and verified {len(_ALL_ATTACK_CLASSES)} attack model implementations.")