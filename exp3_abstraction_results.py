from sklearnex import patch_sklearn
patch_sklearn()
from os.path import exists
import pickle
# from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tabulate import tabulate
import numpy as np
import exp3_abstraction_hypertuning as hp
import exp3_abstraction_loading_data as load
from sklearn.metrics import f1_score, roc_curve, auc, precision_score, recall_score
from joblib import dump

    # Function to evaluate a model
def evaluate_model(model, X_test, y_test):
    model_name = model.__class__.__name__
    # print("Evaluating model:", model_name)

    y_pred = model.predict(X_test)

    if model_name == 'SVC':
        y_pred_proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    f1 = f1_score(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return {"F1 score": f1, "ROC-AUC": roc_auc, "precision": precision, "recall": recall}
def main(saveName = None):
    # Initialize the dictionaries
    results_real = {}
    results_synthetic = {}
    results_synthetic_on_real = {}

    #Models to test
    model_names = ['catboost', 'xgboost', 'RF','KNN', 'SVM' ][::-1]



    # Test each model on the same testing set for real and synthetic data
    for fold_id in range(1, 11):
        for dataset_type in ['real', 'synthetic'][::-1]:
            train_data, validation_data, test_data = load.load_attack_data(
                fold_id, dataset_type)
            X_test, y_test = test_data
            
            
            model_class = hp.models(validation_data, str(dataset_type))
            for model_name in model_names:
                # Load the models and parameters
                print('hyper parameter tuning')
                parameters = model_class.parameters_of(model_name)

                if model_name == 'catboost':
                    model = CatBoostClassifier(**parameters, silent=True)
                elif model_name == 'xgboost':
                    model = xgb.XGBClassifier(**parameters)
                elif model_name == 'RF':
                    model = RandomForestClassifier(**parameters)
                elif model_name == 'SVM':
                    model = SVC(**parameters)
                    
                elif model_name == 'KNN':
                    model = KNeighborsClassifier(**parameters)
                
                print('PREPARING INPUT')
                X_train_, y_train_ = train_data
                xval_, yval_ = validation_data
                X_train, y_train = np.vstack((X_train_,xval_ )), np.hstack((y_train_, yval_))
                
                
                
                # if model_name in ['RF','SVM', 'KNN']:
                #     # Reshape the target variable only for SVM and KNN RF
                #     y_train = np.ravel(y_train.copy())
                #     y_test = np.ravel(y_test.copy())
                
                print('the shape of the training set', X_train.shape)
                
                model.fit(X_train, y_train)
                if saveName is not None:
                    dump(model, f'{model_name}_{dataset_type}_fold{fold_id}_{saveName}.joblib')
                
                
                results = evaluate_model(model, X_test, y_test)

                if dataset_type == 'synthetic':
                    train_data_real, validation_data_real, test_data_real = load.load_attack_data(
                        fold_id, 'real')
                    X_test_real, y_test_real = test_data_real
                    if model_name in ['RF','SVM', 'KNN']:
                        # Reshape the target variable only for SVM and KNN
                        y_test_real = np.ravel(y_test_real.copy())

                    temp_results_synth_real = evaluate_model(
                        model, X_test_real, y_test_real)

                    if model_name not in results_synthetic_on_real:
                        results_synthetic_on_real[model_name] = {
                            key: [] for key in temp_results_synth_real.keys()}
                    for key in temp_results_synth_real.keys():
                        results_synthetic_on_real[model_name][key].append(
                            temp_results_synth_real[key])

                print('table:', model_name,'performance on:', dataset_type)
                print(tabulate([[key, value] for key, value in results.items()], headers=[
                      'metrics', 'result']), '\n')

                if dataset_type == 'synthetic':
                    print('table:', model_name,
                          'trained on synthetic and tested on real:')
                    print(tabulate([[key, value] for key, value in temp_results_synth_real.items(
                    )], headers=['metrics', 'result']), '\n')

                if dataset_type == 'real':
                    if model_name not in results_real:
                        results_real[model_name] = {key: []
                                                    for key in results.keys()}
                    for key in results.keys():
                        results_real[model_name][key].append(results[key])
                elif dataset_type == 'synthetic':
                    if model_name not in results_synthetic:
                        results_synthetic[model_name] = {
                            key: [] for key in results.keys()}
                    for key in results.keys():
                        results_synthetic[model_name][key].append(results[key])

    # Save the results
    with open('results_real.pkl', 'wb') as f:  ##effect of training size on performance
        pickle.dump(results_real, f)

    with open('results_synthetic.pkl', 'wb') as f:
        pickle.dump(results_synthetic, f)

    with open('results_synthetic_on_real.pkl', 'wb') as f:
        pickle.dump(results_synthetic_on_real, f)


def resutls():
    if exists("results_real.pkl") and exists('results_synthetic.pkl') and exists('results_synthetic_on_real.pkl'):
        with open('results_real.pkl', 'rb') as f:
            results_real = pickle.load(f)
        with open('results_synthetic.pkl', 'rb') as f:
            results_synthetic = pickle.load(f)
            
        with open('results_synthetic_on_real.pkl', 'rb') as f:
            results_synthetic_on_real = pickle.load(f)

        return results_real, results_synthetic, results_synthetic_on_real

    else:
        main()  # will save results
        resutls()
