# src/attack_models/base.py
"""
Defines the abstract base class for all attack models.
"""

from abc import ABC, abstractmethod
import pandas as pd

class BaseAttackModel(ABC):
    """
    Abstract Base Class for synthetic attack models.

    Each concrete attack model should inherit from this class and implement
    the `apply` method.
    """

    @property
    @abstractmethod
    def attack_id(self) -> str:
        """Returns the unique identifier for the attack model."""
        pass

    @abstractmethod
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the attack logic to the input data.

        Args:
            data (pd.DataFrame): Input DataFrame, typically representing
                                 energy consumption for one or more consumers
                                 over a specific period (e.g., a month).
                                 Assumes time index and consumers as columns.

        Returns:
            pd.DataFrame: DataFrame with the attack applied, having the same
                          shape, index, and columns as the input data.
        """
        pass
