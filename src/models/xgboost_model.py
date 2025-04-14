# src/models/xgboost_model.py
"""
XGBoost classification model wrapper.
"""
import numpy as np
import xgboost as xgb
from typing import Dict, Any, Optional

from .base_model import BaseModel

class XGBoostModel(BaseModel):
    """Wrapper for the XGBoost Classifier."""

    def _build_model(self) -> xgb.XGBClassifier:
        """Builds the XGBClassifier with stored parameters."""
        default_params = {'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'}
        final_params = {**default_params, **self.params}
        return xgb.XGBClassifier(**final_params)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fits the XGBoost model.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
            **kwargs: Additional arguments passed to XGBClassifier.fit
                      (e.g., eval_set, early_stopping_rounds).
        """
        print(f"Fitting XGBoost model with {X.shape[0]} samples...")
        # XGBoost might need labels flattened or specific types
        y_prepared = y.astype(int) # Ensure integer labels
        try:
            self.model.fit(X, y_prepared, **kwargs)
        except Exception as e:
            print(f"Error during XGBoost fitting: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels using the fitted XGBoost model."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities using the fitted XGBoost model."""
        return self.model.predict_proba(X)
