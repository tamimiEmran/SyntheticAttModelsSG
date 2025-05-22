# src/models/base_model.py
"""
Defines the abstract base class for all machine learning models.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
from .hypertune import models as hypertune_models
import pandas as pd
from hashlib import sha256
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
class BaseModel(ABC):
    """
    Abstract Base Class for machine learning models.

    Ensures a consistent interface for fitting, predicting, and potentially
    saving/loading models.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None, validationTuple: Optional[tuple[np.ndarray, np.ndarray]] = None, hypertuningParamsKeyName: Optional[str] = None):
        """
        Initializes the model.

        Args:
            params (Optional[Dict[str, Any]]): Hyperparameters for the model.
                                              Defaults to None for default params.
        """
        self.params = params if params is not None else {}
        if validationTuple is not None:
            if hypertuningParamsKeyName is None: 
                #hash a random seed from the validation tuple to use as a key
                _x = validationTuple.X
                _y = validationTuple.y
                # Convert to numpy arrays IF not already
                if not isinstance(_x, np.ndarray):
                    _x = _x.to_numpy()
                if not isinstance(_y, np.ndarray):
                    _y = _y.to_numpy()
                # convert to bytes
                _x = _x.tobytes()
                _y = _y.tobytes()

                self.hypertuningParamsKeyName = sha256(_x + _y).digest().hex()
                print(f"Hypertuning key name: {self.hypertuningParamsKeyName}")
            else:
                self.hypertuningParamsKeyName = hypertuningParamsKeyName
            
            assert not np.any(pd.isnull(validationTuple.X.to_numpy())), "NaN in X"
            assert np.all(np.isfinite(validationTuple.X.to_numpy())), "Inf in X"
            assert not np.any(pd.isnull(validationTuple.y)), "NaN in y"
            # assert validationTuple.y is numpy array
            assert isinstance(validationTuple.y, np.ndarray), "y is not numpy array"


            self.validation_tuple = validationTuple.X.to_numpy(), validationTuple.y
            hpKey = self.hypertuningParamsKeyName

            self.hypertuner = hypertune_models(
            validation_tuple= self.validation_tuple,
            real_or_synthetic= hpKey
            )

            self.to_hypertune = True 
        
        else:
            self.to_hypertune = False
        
        
        self.model = self._build_model() # Concrete classes build their specific model
        # move saved hyperparameters to hyperparameters directory

        hyperparameters_dir = os.path.join(PROJECT_ROOT, 'hyperparameters')
        for file in os.listdir(current_dir):
            if file.endswith('.npy') and ("_parameters_" in file):
                os.rename(os.path.join(current_dir, file), os.path.join(hyperparameters_dir, file))



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

    def get_tuned_params(self, name) -> Dict[str, Any]:
        """Returns the tuned parameters after hypertuning."""

        return self.hypertuner.parameters_of(name)
        


