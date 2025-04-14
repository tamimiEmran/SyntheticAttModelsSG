# src/attack_models/implementations.py
"""
Concrete implementations of various synthetic attack models.

Each class inherits from BaseAttackModel and implements a specific attack type.
The logic is adapted from the original `attackTypes.py` and related scripts.
"""

import pandas as pd
import numpy as np
import random
from typing import Union

from .base import BaseAttackModel

# Helper function for random scaling factor (used in several attacks)
def _get_random_alpha(low=0.1, high=0.8):
    return np.random.uniform(low, high)

# --- Concrete Attack Implementations ---

class AttackType0(BaseAttackModel):
    """Attack Type 0: Zero consumption"""
    attack_id = "0"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        # Apply to all columns (consumers)
        for col in modified_data.columns:
            modified_data[col] = 0.0
        return modified_data

class AttackType1(BaseAttackModel):
    """Attack Type 1: Random noise addition"""
    attack_id = "1"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        alpha = _get_random_alpha()
        noise = np.random.uniform(0, alpha, size=data.shape)
        modified_data += noise
        return modified_data

class AttackType2(BaseAttackModel):
    """Attack Type 2: Increase consumption by a random factor"""
    attack_id = "2"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        alpha = _get_random_alpha(low=1.1, high=1.8) # Increase factor
        modified_data *= alpha
        return modified_data

class AttackType3(BaseAttackModel):
    """Attack Type 3: Decrease consumption by a random factor"""
    attack_id = "3"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        alpha = _get_random_alpha() # Decrease factor (0.1 to 0.8)
        modified_data *= alpha
        return modified_data

class AttackType4(BaseAttackModel):
    """Attack Type 4: Add random noise scaled by standard deviation"""
    attack_id = "4"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        std_dev = data.std(axis=0) # Calculate std dev for each consumer
        alpha = _get_random_alpha()

        for col in modified_data.columns:
            noise = np.random.uniform(0, alpha * std_dev[col], size=data.shape[0])
            modified_data[col] += noise
        return modified_data

class AttackType5(BaseAttackModel):
    """Attack Type 5: Add constant value"""
    attack_id = "5"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        alpha = _get_random_alpha()
        modified_data += alpha
        return modified_data

class AttackType6(BaseAttackModel):
    """Attack Type 6: Increase high consumption values"""
    attack_id = "6"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        alpha = _get_random_alpha(low=1.1, high=1.8)
        mean_val = data.mean(axis=0) # Mean per consumer

        for col in modified_data.columns:
            mask = data[col] > mean_val[col]
            modified_data.loc[mask, col] *= alpha
        return modified_data

class AttackType7(BaseAttackModel):
    """Attack Type 7: Decrease low consumption values"""
    attack_id = "7"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        alpha = _get_random_alpha() # Decrease factor
        mean_val = data.mean(axis=0) # Mean per consumer

        for col in modified_data.columns:
            mask = data[col] < mean_val[col]
            modified_data.loc[mask, col] *= alpha
        return modified_data

class AttackType8(BaseAttackModel):
    """Attack Type 8: Shift consumption pattern"""
    attack_id = "8"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        shift_amount = random.randint(1, data.shape[0] - 1)
        # Apply roll to each column
        for col in modified_data.columns:
            modified_data[col] = np.roll(data[col].values, shift=shift_amount)
        return modified_data

class AttackType9(BaseAttackModel):
    """Attack Type 9: Invert consumption pattern"""
    attack_id = "9"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        # Apply flipud (upside down) to each column
        for col in modified_data.columns:
            modified_data[col] = np.flipud(data[col].values)
        return modified_data

class AttackType10(BaseAttackModel):
    """Attack Type 10: Smooth consumption using rolling mean"""
    attack_id = "10"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        window_size = random.randint(2, 7) # Example window size range
        # Apply rolling mean to each column
        for col in modified_data.columns:
            # Using min_periods=1 to handle edges, filling NaNs afterwards
            smoothed = data[col].rolling(window=window_size, min_periods=1).mean()
            modified_data[col] = smoothed.fillna(method='bfill').fillna(method='ffill') # Fill any remaining NaNs
        return modified_data

class AttackType11(BaseAttackModel):
    """Attack Type 11: Introduce random zero periods"""
    attack_id = "11"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        zero_fraction = np.random.uniform(0.05, 0.2) # Fraction of period to zero out
        num_zeros = int(data.shape[0] * zero_fraction)

        for col in modified_data.columns:
            if data.shape[0] > 0: # Check if there are rows to sample from
                 zero_indices = np.random.choice(data.index, size=num_zeros, replace=False)
                 modified_data.loc[zero_indices, col] = 0.0
        return modified_data

class AttackType12(BaseAttackModel):
    """Attack Type 12: Set consumption to the mean"""
    attack_id = "12"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        mean_val = data.mean(axis=0) # Mean per consumer
        for col in modified_data.columns:
            modified_data[col] = mean_val[col]
        return modified_data

class AttackTypeIEEE(BaseAttackModel):
    """Attack Type IEEE: Based on IEEE standard (placeholder implementation)"""
    # Note: The original code's handling of 'ieee' might have been more complex
    # or referred to a specific standard implementation not fully shown.
    # This is a placeholder based on common IEEE attack characteristics (e.g., scaling)
    # It seems the original 'save_attacked_ausgrid' used ID 13 for IEEE? Let's assume scaling.
    attack_id = "ieee" # Also maps to 13 in some contexts

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        modified_data = data.copy()
        # Example: Scale consumption by a factor (similar to Type 2 or 3)
        alpha = _get_random_alpha(low=0.2, high=0.7) # Example scaling factor
        modified_data *= alpha
        return modified_data

# List of all implemented attack classes
_ALL_ATTACK_CLASSES = [
    AttackType0, AttackType1, AttackType2, AttackType3, AttackType4,
    AttackType5, AttackType6, AttackType7, AttackType8, AttackType9,
    AttackType10, AttackType11, AttackType12, AttackTypeIEEE
]
