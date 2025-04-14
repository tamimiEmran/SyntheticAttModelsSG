# src/models/base_model.py
"""
Defines the abstract base class for all machine learning models.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class BaseModel(ABC):
    """
    Abstract Base Class for machine learning models.

    Ensures a consistent interface for fitting, predicting, and potentially
    saving/loading models.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initializes the model.

        Args:
            params (Optional[Dict[str, Any]]): Hyperparameters for the model.
                                              Defaults to None for default params.
        """
        self.params = params if params is not None else {}
        self.model = self._build_model() # Concrete classes build their specific model

    @abstractmethod
    def _build_model(self) -> Any:
        """
        Constructs the underlying model object from the specific library.
        To be implemented by subclasses.

        Returns:
            Any: The instantiated model object (e.g., CatBoostClassifier).
        """
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Trains the model on the provided data.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            **kwargs: Additional arguments specific to the model's fit method
                      (e.g., eval_set for XGBoost/CatBoost).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions (typically class labels) on new data.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predicted class labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for new data.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Array of probabilities for each class (e.g., shape [n_samples, n_classes]).
        """
        pass

    # Optional common methods (can be overridden if needed)
    def get_params(self) -> Dict[str, Any]:
        """Returns the parameters the model was initialized with."""
        return self.params

    def set_params(self, params: Dict[str, Any]):
        """Sets new parameters and rebuilds the model."""
        self.params = params
        self.model = self._build_model()

    # Add save/load methods here if implementing a common serialization strategy
    # def save(self, filepath: str): ...
    # def load(self, filepath: str): ...
