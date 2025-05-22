
from tqdm import tqdm
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()




from os.path import exists
from xgboost import XGBClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from catboost import CatBoostError
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)



# constants
n_trials = 100
n_splits = 3


class models():
    def __init__(self, validation_tuple, real_or_synthetic):
        '''


        Parameters
        ----------
        validation_tuple : tuple of (x, y)
            DESCRIPTION.
        real_or_synthetic : str
            is the data provided real or synthetic? ['real', 'synthetic']

        Returns
        -------
        None. Use the parameters_of method

        '''

        self.validation_tuple = validation_tuple
        self.real_or_synthetic = real_or_synthetic

    def parameters_of(self, model_name):
        '''


        Parameters
        ----------
        model_name : STRING
            the name of the model.
            [catboost, xgboost, RF, SVM, KNN]

        Returns
        -------
        dictionary. The parameters for the model.


        '''
        models_names = {

            'catboost': categorical_boosting,
            'xgboost': xg_boosting,
            'RF': random_forest,
            'SVM': svm_,
            'KNN': knn_
        }

        return models_names[model_name](self.validation_tuple,self.real_or_synthetic )


def categorical_boosting(validation_tuple, real_or_synthetic):
    """
    validation_tuple = (X_val, Y_val)

    """
    if not isinstance(real_or_synthetic, str):
        raise Exception('real_or_synthetic must either be synthetic or real')

    if not exists(f'catboost_parameters_{real_or_synthetic}.npy'):
        parameters = hypertune_catboost(validation_tuple)
        np.save(f'catboost_parameters_{real_or_synthetic}.npy', parameters)
    else:
        parameters = np.load(
            f'catboost_parameters_{real_or_synthetic}.npy', allow_pickle=True).item()

    return parameters


def xg_boosting(validation_tuple, real_or_synthetic):
    """
    validation_tuple = (X_val, Y_val)

    """
    if not exists(f'xgboost_parameters_{real_or_synthetic}.npy'):
        parameters = hypertune_xgboost(validation_tuple)
        np.save(f'xgboost_parameters_{real_or_synthetic}.npy', parameters)
    else:
        parameters = np.load(
            f'xgboost_parameters_{real_or_synthetic}.npy', allow_pickle=True).item()
    
    return parameters

def random_forest(validation_tuple, real_or_synthetic):
    """
    validation_tuple = (X_val, Y_val)
    RF stands for random forest
    """
    if not exists(f'RF_parameters_{real_or_synthetic}.npy'):
        parameters = hypertune_RF(validation_tuple)
        np.save(f'RF_parameters_{real_or_synthetic}.npy', parameters)
    else:
        parameters = np.load(
            f'RF_parameters_{real_or_synthetic}.npy', allow_pickle=True).item()
    
    
    return parameters

def svm_(validation_tuple, real_or_synthetic):
    """
    validation_tuple = (X_val, Y_val)

    """
    if not exists(f'svm_parameters_{real_or_synthetic}.npy'):
        parameters = hypertune_SVM(validation_tuple)
        np.save(f'svm_parameters_{real_or_synthetic}.npy', parameters)
    else:
        parameters = np.load(
            f'svm_parameters_{real_or_synthetic}.npy', allow_pickle=True).item()

    return parameters


def knn_(validation_tuple, real_or_synthetic):
    """
    validation_tuple = (X_val, Y_val)

    """
    if not exists(f'knn_parameters_{real_or_synthetic}.npy'):
        parameters = hypertune_knn(validation_tuple)
        np.save(f'knn_parameters_{real_or_synthetic}.npy', parameters)
    else:
        parameters = np.load(
            f'knn_parameters_{real_or_synthetic}.npy', allow_pickle=True).item()

    return parameters



def hypertune_catboost(validation_tuple):
    X_val, y_val = validation_tuple

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1.0, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-6, 10, log=True),
            "border_count": trial.suggest_int("border_count", 1, 255),
            "random_strength": trial.suggest_float("random_strength", 1e-6, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 1e-6, 10, log=True),
            "task_type": "CPU",
            "loss_function": "Logloss",
            "verbose": False,
            "random_seed": 42,
        }

        # Perform 5-fold cross-validation
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        errors = []

        for train_index, test_index in kfold.split(X_val):
            X_train_cv, X_test_cv = X_val[train_index], X_val[test_index]
            y_train_cv, y_test_cv = y_val[train_index], y_val[test_index]

            # Train the model
            model = CatBoostClassifier(**params)
            try:
                model.fit(X_train_cv, y_train_cv)
            except CatBoostError as e:
                raise optuna.exceptions.TrialPruned()  # fail-safe


            # Make predictions on the test data
            # y_pred_test = model.predict(X_test_cv)
            
            y_pred_test_proba = model.predict_proba(X_test_cv)
            y_pred_test_positive_proba = y_pred_test_proba[:, 1]
            roc_auc = roc_auc_score(y_test_cv, y_pred_test_positive_proba)
            
            
            # Calculate the error on the test data
            error = 1 - roc_auc
            errors.append(error)

        # Calculate the mean error across all folds
        mean_error = np.mean(errors)

        return mean_error

    # Create a study and optimize the hyperparameters
    '''
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    '''
    study = optimize_with_progress(objective, description='hp-tuning catboost')

    # Return the best hyperparameters
    return study.best_params


def hypertune_xgboost(validation_tuple):
    X_val, y_val = validation_tuple

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10, log=True),
            "verbosity": 0,
            "random_state": 42,
        }

        # Perform 5-fold cross-validation
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        errors = []

        for train_index, test_index in kfold.split(X_val):
            X_train_cv, X_test_cv = X_val[train_index], X_val[test_index]
            y_train_cv, y_test_cv = y_val[train_index], y_val[test_index]

            # Train the model
            model = XGBClassifier(**params)
            model.fit(X_train_cv, y_train_cv)

            # Make predictions on the test data
            # y_pred_test = model.predict(X_test_cv)
            y_pred_test_proba = model.predict_proba(X_test_cv)
            y_pred_test_positive_proba = y_pred_test_proba[:, 1]
            roc_auc = roc_auc_score(y_test_cv, y_pred_test_positive_proba)
            
            
            # Calculate the error on the test data
            error = 1 - roc_auc
            errors.append(error)

        # Calculate the mean error across all folds
        mean_error = np.mean(errors)

        return mean_error

    # Create a study and optimize the hyperparameters
    '''
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    '''
    study = optimize_with_progress(objective, description='hp-tuning xgboost')

    # Return the best hyperparameters
    return study.best_params


def hypertune_RF(validation_tuple):
    X_val, y_val = validation_tuple
    y_val = np.ravel(y_val)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "random_state": 42,
            'n_jobs': -1
        }

        # Perform 5-fold cross-validation
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        errors = []

        for train_index, test_index in kfold.split(X_val):
            X_train_cv, X_test_cv = X_val[train_index], X_val[test_index]
            y_train_cv, y_test_cv = y_val[train_index], y_val[test_index]

            # Train the model
            model = RandomForestClassifier(**params)
            model.fit(X_train_cv, y_train_cv)

            # Make predictions on the test data
            # y_pred_test = model.predict(X_test_cv)
            y_pred_test_proba = model.predict_proba(X_test_cv)
            y_pred_test_positive_proba = y_pred_test_proba[:, 1]
            roc_auc = roc_auc_score(y_test_cv, y_pred_test_positive_proba)
            
            
            # Calculate the error on the test data
            error = 1 - roc_auc
            errors.append(error)

        # Calculate the mean error across all folds
        mean_error = np.mean(errors)

        return mean_error

    # Create a study and optimize the hyperparameters
    '''
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    '''
    study = optimize_with_progress(
        objective, description='hp-tuning random forest')

    # Return the best hyperparameters
    return study.best_params


def hypertune_SVM(validation_tuple):
    X_val, y_val = validation_tuple
    y_val = np.ravel(y_val)
    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-5, 1e2, log = True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
            "degree": trial.suggest_int("degree", 1, 5),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "coef0": trial.suggest_float("coef0", 0, 1),
            # "shrinking": trial.suggest_categorical("shrinking", [True, False]),
            "probability": True,
            "random_state": 42,
        }

        # Perform 5-fold cross-validation
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        errors = []

        for train_index, test_index in kfold.split(X_val):
            X_train_cv, X_test_cv = X_val[train_index], X_val[test_index]
            y_train_cv, y_test_cv = y_val[train_index], y_val[test_index]

            # Train the model
            model = SVC(**params)
            model.fit(X_train_cv, y_train_cv)

            # Make predictions on the test data
            # y_pred_test = model.predict(X_test_cv)
            y_pred_test_proba = model.predict_proba(X_test_cv)
            y_pred_test_positive_proba = y_pred_test_proba[:, 1]
            roc_auc = roc_auc_score(y_test_cv, y_pred_test_positive_proba)
            
            
            # Calculate the error on the test data
            error = 1 - roc_auc
            errors.append(error)

        # Calculate the mean error across all folds
        mean_error = np.mean(errors)

        return mean_error

    # Create a study and optimize the hyperparameters
    '''
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    '''
    study = optimize_with_progress(objective, description='hp-tuning SVM')
    # Return the best hyperparameters
    return study.best_params


def hypertune_knn(validation_tuple):
    X_val, y_val = validation_tuple
    y_val = np.ravel(y_val)
    def objective(trial):
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 100),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
            "leaf_size": trial.suggest_int("leaf_size", 10, 50),
            "p": trial.suggest_int("p", 1, 2),
            'n_jobs': -1
        }

        # Perform 5-fold cross-validation
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        errors = []

        for train_index, test_index in kfold.split(X_val):
            X_train_cv, X_test_cv = X_val[train_index], X_val[test_index]
            y_train_cv, y_test_cv = y_val[train_index], y_val[test_index]

            # Train the model
            model = KNeighborsClassifier(**params)
            model.fit(X_train_cv, y_train_cv)

            # Make predictions on the test data
            # y_pred_test = model.predict(X_test_cv)
            y_pred_test_proba = model.predict_proba(X_test_cv)
            y_pred_test_positive_proba = y_pred_test_proba[:, 1]
            roc_auc = roc_auc_score(y_test_cv, y_pred_test_positive_proba)
            
            
            # Calculate the error on the test data
            error = 1 - roc_auc
            errors.append(error)

        # Calculate the mean error across all folds
        mean_error = np.mean(errors)

        return mean_error

    # Create a study and optimize the hyperparameters
    '''
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    '''
    study = optimize_with_progress(objective, description='hp-tuning KNN')

    # Return the best hyperparameters
    return study.best_params



import time

def optimize_with_progress(objective, description='hp-tuning '):
    with tqdm(total=n_trials, desc=description) as progress_bar:
        start_time = time.time()
        
        def callback(study, trial):
            best_value = study.best_value
            progress_bar.update(1)
            elapsed_time = time.time() - start_time
            estimated_time_left = elapsed_time * (n_trials - progress_bar.n) / progress_bar.n
            progress_bar.set_postfix_str(
                f"Best err: {best_value} - Est:{progress_bar.format_interval(estimated_time_left)}",
                refresh=False,
            )
            
            
            
        

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    return study

