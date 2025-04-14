# src/models/rf_model.py
"""
Random Forest classification model wrapper.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional

from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Wrapper for the Scikit-learn RandomForestClassifier."""

    def _build_model(self) -> RandomForestClassifier:
        """Builds the RandomForestClassifier with stored parameters."""
        default_params = {'random_state': 42, 'n_jobs': -1} # Use all available cores
        final_params = {**default_params, **self.params}
        return RandomForestClassifier(**final_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fits the Random Forest model.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            **kwargs: Additional arguments passed to RandomForestClassifier.fit
                      (e.g., sample_weight).
        """
        print(f"Fitting RandomForest model with {X.shape[0]} samples...")
        # Scikit-learn often expects 1D y for classification
        y_prepared = y.ravel()
        try:
            self.model.fit(X, y_prepared, **kwargs)
        except Exception as e:
            print(f"Error during RandomForest fitting: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels using the fitted Random Forest model."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities using the fitted Random Forest model."""
        return self.model.predict_proba(X)
