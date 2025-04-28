
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
from src.data.sgcc_data_preparation import sgcc_wholeConsumer 
from src.models import catboost_model,svm_model, cnn_lstm
from src.evaluation.metrics import Evaluator
from time import time
from copy import deepcopy as cp
# metrics to calculate 

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOT_FILENAME = os.path.join(PROJECT_ROOT)

# SET UP DATA
dataset = sgcc_wholeConsumer()

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

for modelCLS in [catboost_model.CatBoostModel, svm_model.SVMModel, cnn_lstm.CNNLSTMModel][0:1]:
    
    model = modelCLS()  # Initialize the model with hypertuning set to False
    model_name = model.name
    results = ModelResult(name=model_name)
    
    start = time()
    model.fit(dataset.train.X, dataset.train.y)
    end = time()

    training_time = end - start
    results.train_time = training_time
    
    results.train = predictions(y_true=dataset.train.y, y_pred=model.predict_proba(dataset.train.X))
    results.val = predictions(y_true=dataset.val.y, y_pred=model.predict_proba(dataset.val.X))
    results.test = predictions(y_true=dataset.test.y, y_pred=model.predict_proba(dataset.test.X))

    all_results[model_name] = cp(results)
    print(f"Model {model_name} trained in {training_time:.2f} seconds")


aucs = {

}

for model_name, result in all_results.items():
    print(f"Model: {model_name}")
    
    evaluate = Evaluator(
        find_optimal_threshold= True,
        plot_curves = False
    )

    eval_results = evaluate.evaluate(
        y_train_pred = result.train.y_pred,
        y_train_true = result.train.y_true,
        y_val_pred = result.val.y_pred,
        y_val_true = result.val.y_true,
        y_test_pred = result.test.y_pred,
        y_test_true = result.test.y_true

    )

    aucs[model_name] = eval_results.test.auc_roc





    
