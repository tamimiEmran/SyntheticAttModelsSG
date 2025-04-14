# src/models/__init__.py
"""
Machine Learning Models Module

Provides classes for various classification models used for theft detection.
"""
from .base_model import BaseModel
from .catboost_model import CatBoostModel
from .xgboost_model import XGBoostModel
from .rf_model import RandomForestModel
from .svm_model import SVMModel
from .knn_model import KNNModel

# Optional: Define a factory function here if needed later
# def get_model(model_name: str, **kwargs) -> BaseModel:
#     ...

__all__ = [
    "BaseModel",
    "CatBoostModel",
    "XGBoostModel",
    "RandomForestModel",
    "SVMModel",
    "KNNModel",
]
