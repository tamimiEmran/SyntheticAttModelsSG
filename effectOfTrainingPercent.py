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
from applyAttacks import PCA_fullDataExamples
import matplotlib.pyplot as plt
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

pcaFoldExamples = 1#[ PCA_fullDataExamples(fold_id)  for fold_id in range(1, 11) ]
def results():
    model_names = ['catboost', 'xgboost', 'RF','KNN', 'SVM' ]
    all_results = {}
    for fold_id in range(1, 11):
        all_results[fold_id] = {}
        train, validation, test = pcaFoldExamples[fold_id - 1]
        
        train_examples_full = np.vstack((train[0], validation[0]))
        train_labels_full = np.vstack((train[1], validation[1]))
        
        length = len(train_examples_full)
        print('original train examples size:', train_examples_full.shape)
        permuted_indices = np.random.permutation(length)
        train_examples_full = train_examples_full[permuted_indices]
        train_labels_full = train_labels_full[permuted_indices]
        
        test_examples, test_labels = test
        
        
        for percentage in range(10, 110, 10):
            all_results[fold_id][percentage] = {}
            examples_to_take = int(percentage/100 * length)
            
            print(percentage, examples_to_take)
            
            train_examples = train_examples_full[:examples_to_take ]
            print('New train examples size:', train_examples.shape)
            
            train_labels = train_labels_full[:examples_to_take ]
            
            
            
            for model_name in model_names:
                # Load the models and parameters
    
                if model_name == 'catboost':
                    model = CatBoostClassifier(silent=True)
                elif model_name == 'xgboost':
                    model = xgb.XGBClassifier()
                elif model_name == 'RF':
                    model = RandomForestClassifier()
                elif model_name == 'SVM':
                    model = SVC( probability=True)
                elif model_name == 'KNN':
                    model = KNeighborsClassifier()

                if model_name in ['RF','SVM', 'KNN']:
                    # Reshape the target variable only for SVM and KNN RF
                    train_labels = np.ravel(train_labels)
                    test_labels = np.ravel(test_labels)
                
            
                model.fit(train_examples, train_labels)
                results = evaluate_model(model, test_examples,test_labels)
                
                print(model_name, results, sep = '\n')
                
                all_results[fold_id][percentage][model_name] = results
        
    return all_results
plt.rcParams['figure.figsize'] = [10, 6]  # width, height in inches
plt.rcParams['figure.dpi'] = 100 
def plot(all_results = None):
 # Resolution in dots per inch
    # Load the results from the file
    if all_results is None:
        with open("all_results.pkl", "rb") as f:
            all_results = pickle.load(f)
        
    model_names = ['catboost', 'xgboost', 'RF', 'KNN', 'SVM']
    metrics = ["F1 score", "ROC-AUC", "precision", "recall"]
    percentages = list(range(10, 110, 10))
    
    # Initialize a dictionary to store mean and standard deviation values
    mean_std_values = {metric: {model_name: [] for model_name in model_names} for metric in metrics}
    
    for metric in metrics:
        for model_name in model_names:
            for percentage in percentages:
                values = [all_results[fold_id][percentage][model_name][metric] for fold_id in range(1, 11)]
                mean_std_values[metric][model_name].append((np.mean(values), np.std(values)))
    
    # Plot the results
    for metric in metrics:
        fig, ax = plt.subplots()
        
        for i, model_name in enumerate(model_names):
            means, stds = zip(*mean_std_values[metric][model_name])
            ax.plot(percentages, means, label=model_name)
            ax.fill_between(percentages, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.3)
        
        ax.set_title(metric)
        ax.set_xlabel("Percentage of Training Data")
        ax.set_ylabel("Scores")
        ax.legend()
    
    plt.show()
            
# all_results = results()
# with open("all_results.pkl", "wb") as f:
#     pickle.dump(all_results, f)
    
    

plot()