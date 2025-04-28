# src/models/catboost_model.py
"""
CatBoost classification model wrapper.
"""
import numpy as np
from catboost import CatBoostClassifier
from typing import Dict, Any, Optional

from .base_model import BaseModel
class CatBoostModel(BaseModel):
    name = "catboost"
    """Wrapper for the CatBoost Classifier."""
    def _build_model(self) -> CatBoostClassifier:
        """Builds the CatBoostClassifier with stored parameters."""
        # Ensure common params like random_state and verbosity are handled
        if self.to_hypertune is None:
            self.to_hypertune = False
        
        to_hypertune = self.to_hypertune
        if to_hypertune: 
            params = super().hypertune("catboost")
            default_params = {'random_state': 42, 'verbose': False, "loss_function":'Logloss'}
            final_params = {**default_params, **params} 
            return CatBoostClassifier(**final_params)

        else: 
            default_params = {'random_state': 42, 'verbose': False, "loss_function":'Logloss'}
            final_params = {**default_params, **self.params} # User params override defaults
            return CatBoostClassifier(**final_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fits the CatBoost model.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            **kwargs: Additional arguments passed to CatBoostClassifier.fit
                      (e.g., eval_set, early_stopping_rounds).
        """
        print(f"Fitting CatBoost model with {X.shape[0]} samples...")
        # CatBoost usually handles label types well, no explicit flattening needed
        try:
            self.model.fit(X, y, **kwargs)
        except Exception as e:
            print(f"Error during CatBoost fitting: {e}")
            # Consider re-raising or logging more details
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels using the fitted CatBoost model."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities using the fitted CatBoost model."""
        return self.model.predict_proba(X)