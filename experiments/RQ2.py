
import os
import sys
import numpy as np
import pickle
from collections import defaultdict
from dataclasses import dataclass
#add project root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Imports ---
from src.data.sgcc_data_preparation import sgcc_monthly
from src.models import catboost_model,svm_model, xgboost_model, knn_model, rf_model
from src.utils.evaluation import evaluate_aggregated, evaluate_monthly, _default_metrics, compute_optimal_threshold
from time import time
from copy import deepcopy as cp

# SET UP DATA
dataset = sgcc_monthly()

@dataclass
class predictions:
    y_true: np.ndarray = None
    y_pred: np.ndarray = None


@dataclass
class ModelResult:
    name: str
    train_time: float = 0.0
    train: predictions = None
    val: predictions = None
    test: predictions = None

all_results = {

}

modelCLS = catboost_model.CatBoostModel

model = modelCLS()
model_name = model.name
results = ModelResult(name= "monthly_" + model_name)

start = time()
model.fit(dataset.train.X, dataset.train.y)
end = time()

training_time = end - start
results.train_time = training_time

results.train = predictions(y_true=dataset.train.y, y_pred=model.predict_proba(dataset.train.X))
results.val = predictions(y_true=dataset.val.y, y_pred=model.predict_proba(dataset.val.X))
results.test = predictions(y_true=dataset.test.y, y_pred=model.predict_proba(dataset.test.X))

all_results[model_name] = cp(results)


#%% 
p_val = results.val.y_pred[:, 1]  # Get the probabilities for the positive class
thr_val = compute_optimal_threshold(
    results.val.y_true, p_val, metric="f1"
)
metrics = _default_metrics(threshold=thr_val)


scores_month = evaluate_monthly(
    model, dataset.dataframe.test, dataset.labels.test,
    metrics=metrics
)

scores_consumer = evaluate_aggregated(
    model, dataset.dataframe.test, dataset.labels.test,
    metrics=metrics
)

print("Monthly-level scores:", scores_month)
print("Consumer-level scores:", scores_consumer)

