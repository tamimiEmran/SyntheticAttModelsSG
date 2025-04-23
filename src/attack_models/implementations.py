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
import numpy as np

import pandas as pd
from collections import defaultdict
from typing import Dict, List, Union
import pandas as pd
from typing import Union

def redistribute_consumption(
    df: pd.DataFrame,
    ratio: float = 0.4,
    in_place: bool = False
) -> pd.DataFrame:
    """
    1. Build a Series of each user's total consumption in `df`.
    2. For every (row-)user *i*, find another user *j* whose total
       consumption `total[j]` is **≤ ratio * total[i]`**, but is
       as large as possible among the candidates.
    3. Replace the *values* of row *i* with row *j*'s values
       (row copy, same shape).
    4. If no such *j* exists, zero-out row *i*.

    Parameters
    ----------
    df : pd.DataFrame
        Slice of shape (n_users, n_days) you obtained from `result`.
    ratio : float, default 0.4
        The cut-off fraction (`total[j] ≤ ratio * total[i]`).
    in_place : bool, default False
        • False → work on a copy and return it (original stays intact).
        • True  → modify `df` directly and return it (handy in pipelines).

    Returns
    -------
    pd.DataFrame
        The slice after row replacements.
    """
    # ------------------------------------------------------------------
    # 1. Total kWh per user
    totals = df.sum(axis=1)           # Series indexed by user id
    if totals.empty:                  # safety for empty slices
        return df

    # copy or view?
    target = df if in_place else df.copy()

    # cache original rows so replacements all reference the *original* values
    original_rows = df.to_dict(orient="index")

    # ------------------------------------------------------------------
    # 2-4. Row-by-row replacement
    for user, my_sum in totals.items():
        # find candidates strictly below me but ≥ 0 (could equal if ratio ≥ 1)
        mask = (totals <= my_sum * ratio) & (totals <  my_sum)
        candidates = totals[mask]

        if not candidates.empty:
            donor = candidates.idxmax()              # biggest feasible donor
            target.loc[user] = original_rows[donor]  # 3. replace values
        else:
            target.loc[user] = 0.0                   # 4. no donor → zeros

    return target

def build_group_period_slices(
    data: pd.DataFrame,
    users_per_group: int = 100,
    period: str = "M",                 # "M" = month-year, "D" = exact date, etc.
    orientation: str = "group-first"   # or "period-first"
) -> Dict[int, Dict[pd.Period, pd.DataFrame]]:
    """
    Split a wide time-series DataFrame into (group_id ▸ period ▸ DataFrame) slices.

    Parameters
    ----------
    data : pd.DataFrame
        Rows are consumers; columns are daily timestamps.
    users_per_group : int, default 100
        Number of consumers in each group.
    period : str, default "M"
        Pandas offset alias to group columns into periods.
        • "M" → month-year  (2014-05, 2014-06, …)
        • "D" → keep exact dates (one period per column)
        • any valid `to_period()` freq works ("Q", "Y", etc.).
    orientation : {"group-first", "period-first"}, default "group-first"
        Return structure:
        • "group-first": result[group_id][period]  -> DataFrame
        • "period-first": result[period][group_id] -> DataFrame

    Returns
    -------
    dict
        Nested dictionaries of DataFrame slices.
    """
    # 1 ── build consumer groups ------------------------------------------------
    n = len(data)
    base_groups = n // users_per_group
    group_ids: Dict[int, List[int]] = {
        gid: data.index[gid*users_per_group : (gid+1)*users_per_group].tolist()
        for gid in range(base_groups)
    }
    # tack any leftovers onto the previous group
    if n % users_per_group:
        leftovers = data.index[base_groups*users_per_group : ].tolist()
        group_ids.setdefault(base_groups - 1, []).extend(leftovers)

    # 2 ── convert the column index to the requested Period granularity ---------
    periods = data.columns.to_series().dt.to_period(period)
    unique_periods = periods.unique()

    # 3 ── carve out DataFrame slices ------------------------------------------
    make_store = lambda: defaultdict(dict)  # convenience
    if orientation == "group-first":
        result: Dict[int, Dict[pd.Period, pd.DataFrame]] = make_store()
        for gid, ridx in group_ids.items():
            gdf = data.loc[ridx]                  # subset rows
            for p in unique_periods:
                result[gid][p] = gdf.loc[:, periods == p]
    else:  # period-first
        result: Dict[pd.Period, Dict[int, pd.DataFrame]] = make_store()
        for p in unique_periods:
            cidx = periods == p                   # subset columns once
            for gid, ridx in group_ids.items():
                result[p][gid] = data.loc[ridx, cidx]

    return result


def redistribute_consumption_fast(df: pd.DataFrame, ratio: float = 0.4,
                                  in_place: bool = False) -> pd.DataFrame:
    """
    Same semantics as your original function but *O(n log n)* and 100 %
    NumPy inside the hot path.
    """
    if df.empty:
        return df

    tgt = df if in_place else df.copy()
    vals  = tgt.values                 # (n_users, n_cols) float64
    sums  = vals.sum(axis=1)           # (n_users,)
    
    idx_sorted       = np.argsort(sums)               # ascending totals
    sorted_totals    = sums[idx_sorted]               # view
    thresholds       = sums * ratio                   # each user’s limit

    # donor position for every user (binary search on the sorted totals)
    pos = np.searchsorted(sorted_totals, thresholds, side="right") - 1
    donors = idx_sorted[pos]                          # candidate indices

    # users whose own total is <= ratio*total (self or equal) need a new donor
    no_donor = (pos < 0) | (sorted_totals[pos] >= sums)
    
    # --- assemble the output array -----------------------------------
    donor_rows = vals[donors]                         # (n_users, n_cols)
    vals[:]    = donor_rows                           # broadcast copy
    vals[no_donor] = 0.0                              # zero‑out where needed
    return tgt

def apply_attack12(data: pd.DataFrame,
                   group_size: int,
                   period: str,
                   ratio: float) -> pd.DataFrame:
    out = data.copy(deep=True)                       # full copy once
    periods = out.columns.to_series().dt.to_period(period)
    period_codes, _ = pd.factorize(periods, sort=False)
    unique_periods = np.unique(period_codes)

    n = len(out)
    for g_start in range(0, n, group_size):
        rows = slice(g_start, min(g_start + group_size, n))
        for p in unique_periods:
            cols_mask = period_codes == p            # Boolean 1‑D mask

            # --- work on a *copy* of the block -------------------------
            block = out.iloc[rows, cols_mask].copy()
            redistribute_consumption_fast(block, ratio=ratio, in_place=True)

            # --- write the modified values back ------------------------
            out.iloc[rows, cols_mask] = block.values

    return out



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
            # If the series is too short, return it all zeroed out
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
    """Attack Type 3: Apply per-cell random scaling"""
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
    """
    Attack Type 10: Reverses time order within temporal segments
    
    This attack model reverses the order of consumption values within each time segment
    (week for daily data, day for higher-frequency data) while maintaining the overall
    structure. This simulates a sophisticated theft technique that manipulates consumption
    patterns to potentially take advantage of time-of-use pricing.
    """
    attack_id = "10"

    def _validate_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and converts columns to DatetimeIndex if needed.
        
        Args:
            data: DataFrame with potential datetime columns
            
        Returns:
            DataFrame with datetime columns
            
        Raises:
            ValueError: If columns cannot be converted to datetime
        """
        if not isinstance(data.columns, pd.DatetimeIndex):
            try:
                data = data.copy()
                data.columns = pd.to_datetime(data.columns)
                return data
            except Exception as e:
                raise ValueError(f"Column conversion to datetime failed: {e}")
        return data
    
    def _determine_data_frequency(self, data: pd.DataFrame) -> str:
        """
        Determines if data has daily frequency (SGCC) or higher frequency (Ausgrid).
        
        Args:
            data: DataFrame with datetime columns
            
        Returns:
            String indicating data type: "daily" or "high_frequency"
        """
        if data.empty or len(data.columns) < 2:
            return "unknown"
            
        # Sample the first day and count readings
        first_day = data.columns[0].date()
        columns_for_first_day = sum(col.date() == first_day for col in data.columns)
        
        return "daily" if columns_for_first_day <= 1 else "high_frequency"
    

    
    def _get_week_groups(self, columns: pd.DatetimeIndex) -> dict:
        """
        Groups column indices by ISO calendar week.
        Week key format: YYYY-Www where YYYY is *ISO year*.
        """
        week_groups: dict[str, list[int]] = {}
        
        for idx, ts in enumerate(columns):
            iso = ts.isocalendar()          # (iso_year, iso_week, iso_day)
            week_key = f"{iso[0]}-W{iso[1]:02d}"
            week_groups.setdefault(week_key, []).append(idx)
            
        return week_groups

    def _get_day_groups(self, columns: pd.DatetimeIndex) -> dict:
        """
        Groups column indices by day for high-frequency data.
        
        Args:
            columns: DatetimeIndex of columns
            
        Returns:
            Dictionary mapping days to lists of column indices
        """
        day_groups = {}
        for i, col in enumerate(columns):
            day = col.date()
            if day not in day_groups:
                day_groups[day] = []
            day_groups[day].append(i)
        return day_groups
    
    def _reverse_segment(self, data: pd.DataFrame, indices: list) -> None:
        """
        Reverses values within a segment (week or day) for all customers at once.
        
        Args:
            data: DataFrame to modify (in-place)
            indices: List of column indices to reverse
        """
        if len(indices) <= 1:
            return  # Nothing to reverse
            
        # Sort indices to ensure chronological order
        indices.sort(key=lambda i: data.columns[i])
        # Get column names for this segment
        segment_cols = [data.columns[i] for i in indices]
        # Reverse and assign back in one operation
        data.loc[:, segment_cols] = data.loc[:, segment_cols].iloc[:, ::-1].values
    
    def _process_daily_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Processes data with daily frequency (SGCC), reversing weeks.
        
        Args:
            data: DataFrame with daily readings
            
        Returns:
            DataFrame with reversed readings within each week
        """
        result = data.copy()
        week_groups = self._get_week_groups(data.columns)
        
        # Process each week
        for week_id, indices in week_groups.items():
            self._reverse_segment(result, indices)
            
        return result
    
    def _process_high_frequency_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Processes data with high frequency (Ausgrid), reversing days.
        
        Args:
            data: DataFrame with high-frequency readings
            
        Returns:
            DataFrame with reversed readings within each day
        """
        result = data.copy()
        day_groups = self._get_day_groups(data.columns)
        
        # Process each day
        for day, indices in day_groups.items():
            self._reverse_segment(result, indices)
            
        return result
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the attack by reversing time order within appropriate segments.
        
        Args:
            data: Input DataFrame (customers as rows, datetime as columns)
            
        Returns:
            DataFrame with reversed temporal segments
            
        Raises:
            ValueError: If datetime conversion fails
        """
        try:
            # Validate and prepare data
            data = self._validate_datetime_columns(data)
            
            # Determine data frequency and process accordingly
            frequency = self._determine_data_frequency(data)
            
            if frequency == "daily":
                return self._process_daily_data(data)
            elif frequency == "high_frequency":
                return self._process_high_frequency_data(data)
            else:
                # Unknown frequency - log warning and return original
                print("Warning: Unable to determine data frequency. Returning original data.")
                return data.copy()
                
        except Exception as e:
            print(f"Error in AttackType10: {e}")
            # Return original data on error to avoid breaking pipeline
            return data.copy()
        

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
    Attack Type 12: Swap consumption profiles based on sums within weeks.
    """
    attack_id = "12"
    ratio = ATTACK_CONSTANTS["attack12Factor"]
    sgcc_group_size = ATTACK_CONSTANTS["sgcc_group_size"] # 500
    ausgrid_group_size = ATTACK_CONSTANTS["ausgrid_group_size"] # 25

    def _validate_datetime_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and converts columns to DatetimeIndex if needed.
        
        Args:
            data: DataFrame with potential datetime columns
            
        Returns:
            DataFrame with datetime columns
            
        Raises:
            ValueError: If columns cannot be converted to datetime
        """
        if not isinstance(data.columns, pd.DatetimeIndex):
            try:
                data = data.copy()
                data.columns = pd.to_datetime(data.columns)
                return data
            except Exception as e:
                raise ValueError(f"Column conversion to datetime failed: {e}")
        return data
    def _determine_data_frequency(self, data: pd.DataFrame) -> str:
        """
        Determines if data has daily frequency (SGCC) or higher frequency (Ausgrid).
        
        Args:
            data: DataFrame with datetime columns
            
        Returns:
            String indicating data type: "daily" or "high_frequency"
        """
        if data.empty or len(data.columns) < 2:
            return "unknown"
            
        # Sample the first day and count readings
        first_day = data.columns[0].date()
        columns_for_first_day = sum(col.date() == first_day for col in data.columns)
        
        return "daily" if columns_for_first_day <= 1 else "high_frequency"
       

        
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._validate_datetime_columns(data)
        freq = self._determine_data_frequency(data)
        
        if freq == "daily":
            gsize, period = self.sgcc_group_size, "M"     # month buckets
        elif freq == "high_frequency":
            gsize, period = self.ausgrid_group_size, "D"  # daily buckets
        else:
            raise ValueError("Unknown data frequency")

        return apply_attack12(
            data,
            group_size = gsize,
            period     = period,
            ratio      = self.ratio
        )



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

