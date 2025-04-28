# src/models/base_model.py
"""
Defines the abstract base class for all machine learning models.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
from .hypertune import models as hypertune_models

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
                self.hypertuningParamsKeyName = str(hash(validationTuple[0].tobytes() + validationTuple[1].tobytes()))
            else:
                self.hypertuningParamsKeyName = hypertuningParamsKeyName
            self.params = self.hypertune()


            validation_tuple = self.validationTuple
            hpKey = self.hypertuningParamsKeyName

            hypertuner = hypertune_models(
            validation_tuple= validation_tuple,
            real_or_synthetic= hpKey
            )

            self.to_hypertune = True 
        
        else:
            self.to_hypertune = False
        
        
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

    def get_tuned_params(self, name) -> Dict[str, Any]:
        """Returns the tuned parameters after hypertuning."""

        return self.hypertuner.parameters_of(name)
        


