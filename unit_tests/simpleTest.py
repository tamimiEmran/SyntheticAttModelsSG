#%% imports
# Add project root to sys.path to allow importing project modules
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import random
from src.attack_models import get_attack_model, list_available_attacks
from src.data.loader import load_sgcc_data, load_ausgrid_data
import pandas as pd
#%%
sgcc_data, sgcc_labels = load_sgcc_data()
ausgrid_data = load_ausgrid_data()

#%% 
def return_attacked_sgcc_data(attack_id: str) -> pd.DataFrame:
    """
    Apply an attack model to the SGCC dataset and return the attacked data.
    
    Args:
        attack_id: ID of the attack model to apply

    """
    # Initialize the attack model
    attack_model = get_attack_model(attack_id)
    
    # Apply the attack to the SGCC data
    attacked_data = attack_model.apply(sgcc_data.copy())
    
    return attacked_data

def return_attacked_ausgrid_data(attack_id: str) -> pd.DataFrame:
    """
    Apply an attack model to the Ausgrid dataset and return the attacked data.
    
    Args:
        attack_id: ID of the attack model to apply

    """
    # Initialize the attack model
    attack_model = get_attack_model(attack_id)
    
    # Apply the attack to the Ausgrid data
    attacked_data = attack_model.apply(ausgrid_data.copy())
    
    return attacked_data

#%% attack_visualizer.py
"""A minimal utility for sanity-checking synthetic attack outputs.

Updates (2025-04-21 — ISO-week aware)
-------------------------------------
* **ISO-week alignment** - `check_attack10` now *guarantees* that the plotted
  window matches *exact* ISO weeks (Mon-Sun) so the reversal effect of
  ``AttackType10`` is crystal-clear.
* Adds private helpers ``_iso_week_window`` and ``_random_iso_week_window``.
* Automatically labels the week in plot titles (e.g. *2024-W05*).

Example
~~~~~~~
>>> vis.check_attack10(show_aggregate=True)   # one-liner sanity check
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AttackVisualizer:
    """Visual comparison between original and attacked consumption profiles."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        original_df: pd.DataFrame,
        attacked_df: pd.DataFrame,
        rng_seed: Optional[int] = None,
    ) -> None:
        if not original_df.index.equals(attacked_df.index):
            raise ValueError("Row indices of original and attacked data must match")
        if not original_df.columns.equals(attacked_df.columns):
            raise ValueError("Column DateTime indices must be identical")
        self.orig = original_df
        self.att = attacked_df
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------
    def _sample_meters(self, n: int) -> Sequence[int]:
        return self._rng.choice(self.orig.index, size=n, replace=False).tolist()

    def _random_window(self, length: pd.Timedelta) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Return a *random contiguous* window of given *length*."""
        cols = self.orig.columns
        if len(cols) == 0:
            raise ValueError("DataFrame has no columns")
        start_bound = cols[0]
        stop_bound = cols[-1] - length
        if stop_bound <= start_bound:
            return start_bound, cols[-1]
        delta_seconds = int((stop_bound - start_bound).total_seconds())
        offset = pd.Timedelta(seconds=int(self._rng.integers(delta_seconds)))
        start = start_bound + offset
        return start, start + length

    # ISO-WEEK -----------------------------------------------------------------
    def _iso_week_window(self, reference_ts: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
        """Return start/end covering **exactly** the ISO week of *reference_ts*.

        Also returns the canonical *YYYY-Www* string for labelling.
        """
        cols = self.orig.columns
        iso_ref = reference_ts.isocalendar()  # (year, week, day)
        week_mask = [
            (ts.isocalendar().year == iso_ref.year and ts.isocalendar().week == iso_ref.week)
            for ts in cols
        ]
        week_cols = cols[week_mask]
        if len(week_cols) == 0:
            raise ValueError("Selected ISO week not present in columns")
        week_key = f"{iso_ref.year}-W{iso_ref.week:02d}"
        return week_cols[0], week_cols[-1], week_key

    def _random_iso_week_window(self) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
        ts = self.orig.columns[self._rng.integers(len(self.orig.columns))]
        return self._iso_week_window(ts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plot_meter(
        self,
        meter_id: Optional[int] = None,
        *,
        start: Optional[str | pd.Timestamp] = None,
        end: Optional[str | pd.Timestamp] = None,
        title_suffix: str = "",
        figsize: tuple[int, int] = (12, 4),
        alpha_attacked: float = 0.75,
        auto_window_days: int = 14,
    ) -> None:
        if meter_id is None:
            meter_id = self._sample_meters(1)[0]
        if start is None and end is None:
            start, end = self._random_window(pd.Timedelta(days=auto_window_days))
        # Slice
        orig_series = self.orig.loc[meter_id].loc[start:end]
        att_series = self.att.loc[meter_id].loc[start:end]
        # Draw
        plt.figure(figsize=figsize)
        plt.plot(orig_series.T, label="Original", linewidth=1)
        plt.plot(att_series.T, label="Attacked", linewidth=1, alpha=alpha_attacked)
        plt.title(f"Meter {meter_id} {title_suffix}".strip())
        plt.xlabel("Time")
        plt.ylabel("Consumption")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def quick_check(
        self,
        n: int = 3,
        *,
        auto_window_days: int = 14,
    ) -> None:
        start, end = self._random_window(pd.Timedelta(days=auto_window_days))
        for m in self._sample_meters(n):
            self.plot_meter(meter_id=m, start=start, end=end)

    # ------------------------------------------------------------------
    # Attack-specific helpers
    # ------------------------------------------------------------------
    def check_attack10(
        self,
        meter_id: Optional[int] = None,
        *,
        week_start: Optional[str | pd.Timestamp] = None,
        show_aggregate: bool = False,
    ) -> None:
        """Visualise one full ISO week to verify reversal of Attack 10."""
        # Determine window ------------------------------------------------
        if week_start is None:
            start, end, week_key = self._random_iso_week_window()
        else:
            start, end, week_key = self._iso_week_window(pd.to_datetime(week_start))

        # Plot meter trace -----------------------------------------------
        self.plot_meter(
            meter_id=meter_id,
            start=start,
            end=end,
            title_suffix=f"- Attack 10 (ISO {week_key})",
        )

        # Optional aggregate view ----------------------------------------
        if show_aggregate:
            m = meter_id or self._sample_meters(1)[0]
            orig_week = self.orig.loc[m].loc[start:end].resample("D").sum()
            att_week = self.att.loc[m].loc[start:end].resample("D").sum()
            plt.figure(figsize=(8, 3))
            plt.plot(orig_week.values, marker="o", label="Original daily total")
            plt.plot(att_week.values, marker="o", label="Attacked daily total")
            plt.title(f"Meter {m} - Daily totals (should read reversed)")
            plt.xticks(range(len(orig_week)), [d.strftime("%a") for d in orig_week.index])
            plt.ylabel("kWh / day")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def check_attack12(
        self,
        meter_a: Optional[int] = None,
        meter_b: Optional[int] = None,
        *,
        start: Optional[str | pd.Timestamp] = None,
        end: Optional[str | pd.Timestamp] = None,
    ) -> None:
        if meter_a is None or meter_b is None:
            meter_a, meter_b = self._sample_meters(2)
        if start is None and end is None:
            start, end = self._random_window(pd.Timedelta(days=14))
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for ax, m in zip(axes, (meter_a, meter_b)):
            ax.plot(self.orig.loc[m].loc[start:end].T, label="Original", linewidth=1)
            ax.plot(self.att.loc[m].loc[start:end].T, label="Attacked", linewidth=1, alpha=0.75)
            ax.set_title(f"Meter {m}")
            ax.legend()
        fig.suptitle("Attack 12 - consumption swap check")
        plt.tight_layout()
        plt.show()

# %%

if __name__ == "__main__":
    # Example usage
    attack_id = 12
    attacked_sgcc_data = return_attacked_sgcc_data(attack_id)
    attacked_ausgrid_data = return_attacked_ausgrid_data(attack_id)
    
    changed_any = not attacked_sgcc_data.equals(sgcc_data)
    print("Did the attack modify *anything*? ->", changed_any)

    changed_any = not attacked_ausgrid_data.equals(ausgrid_data)
    print("Did the attack modify *anything*? ->", changed_any)


    # Visualize the attack
    viz_sgcc = AttackVisualizer(sgcc_data, attacked_sgcc_data)
    viz_sgcc.check_attack12()

    viz_ausgrid = AttackVisualizer(ausgrid_data, attacked_ausgrid_data)
    viz_ausgrid.check_attack12()
    

# %%
