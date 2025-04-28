# src/models/svm_model.py
"""
Support Vector Machine (SVM) classification model wrapper.
"""
import numpy as np
from sklearn.svm import SVC
from typing import Dict, Any, Optional

from .base_model import BaseModel

class SVMModel(BaseModel):
    """Wrapper for the Scikit-learn SVC (Support Vector Classifier)."""
    name = "SVM"
    def _build_model(self) -> SVC:
        """Builds the SVC with stored parameters."""
        # Ensure probability=True if predict_proba will be used
        if self.to_hypertune:
            params = super().hypertune("SVM")
            default_params = {'random_state': 42, 'probability': True, 'kernel': 'linear'}
            final_params = {**default_params, **params}
            return SVC(**final_params)
        else: 
            default_params = {'random_state': 42, 'probability': True, 'kernel': 'linear'}
            final_params = {**default_params, **self.params}
            return SVC(**final_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fits the SVM model.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            **kwargs: Additional arguments passed to SVC.fit
                      (e.g., sample_weight).
        """
        print(f"Fitting SVM model with {X.shape[0]} samples...")
        # Scikit-learn often expects 1D y for classification
        y_prepared = y.ravel()
        try:
            self.model.fit(X, y_prepared, **kwargs)
        except Exception as e:
            print(f"Error during SVM fitting: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels using the fitted SVM model."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities using the fitted SVM model.
        Requires probability=True during initialization.
        """
        if not self.model.probability:
            raise AttributeError("predict_proba is not available when probability=False")
        return self.model.predict_proba(X)


