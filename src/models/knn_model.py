# src/models/knn_model.py
"""
K-Nearest Neighbors (KNN) classification model wrapper.
"""
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, Any, Optional

from .base_model import BaseModel

class KNNModel(BaseModel):
    """Wrapper for the Scikit-learn KNeighborsClassifier."""

    def _build_model(self) -> KNeighborsClassifier:
        """Builds the KNeighborsClassifier with stored parameters."""
        if self.to_hypertune:
            params = self.hypertuner.parameters_of("KNN")
            default_params = {'n_jobs': -1}
            final_params = {**default_params, **params} # User params override defaults
            return KNeighborsClassifier(**final_params)
        
        else:
            default_params = {'n_jobs': -1} # Use all available cores
            final_params = {**default_params, **self.params}
            return KNeighborsClassifier(**final_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fits the KNN model. (Note: KNN is instance-based, 'fit' mainly stores data).

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            **kwargs: Additional arguments (ignored by KNeighborsClassifier.fit).
        """
        print(f"Fitting KNN model with {X.shape[0]} samples...")
        # Scikit-learn often expects 1D y for classification
        y_prepared = y.ravel()
        try:
            self.model.fit(X, y_prepared) # kwargs are typically not used by KNN fit
            if kwargs:
                print(f"Warning: kwargs provided but ignored by KNN fit: {kwargs.keys()}")
        except Exception as e:
            print(f"Error during KNN fitting: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels using the fitted KNN model."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities using the fitted KNN model."""
        return self.model.predict_proba(X)
