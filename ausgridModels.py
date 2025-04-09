import exp3_abstraction_hypertuning as hp
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
from sklearn.metrics import f1_score, roc_curve, auc, precision_score, recall_score
from collections import defaultdict

import dataset_setup
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


model_names = ['catboost', 'xgboost', 'RF','KNN', 'SVM' ]
dataclass = dataset_setup.data_loader()
results_all = defaultdict(lambda: defaultdict(list))



### test_models
for model_name in model_names:
    for foldId in range(3):
        train, test = dataclass.train_test_dailyExamples(foldId)
        
        
        
        x_train, y_train = train
        x_test, y_test = test
        
        trainingSize = x_train.shape[0]
        validationSize = int(trainingSize * 0.01)
        
        # hyperParameters = hp.models( (x_train[:validationSize], y_train[:validationSize]) , 'ausgrid')
        # parameters = hyperParameters.parameters_of(model_name)
        
        parameters = {}
        

        if model_name == 'catboost':
            model = CatBoostClassifier(**parameters, silent=True)
        elif model_name == 'xgboost':
            model = xgb.XGBClassifier(**parameters)
        elif model_name == 'RF':
            model = RandomForestClassifier(**parameters, max_features=1.0)
        elif model_name == 'SVM':
            model = SVC(**parameters)
            
        elif model_name == 'KNN':
            model = KNeighborsClassifier(**parameters)
        
        model.fit(x_train, y_train)
        
        results = evaluate_model(model, x_test, y_test)
        ### results are a dictionary as follows: {"F1 score": f1, "ROC-AUC": roc_auc, "precision": precision, "recall": recall}
        # Append the results to results_all dictionary
        
        print('table:', model_name, 'fold', foldId)
        print(tabulate([[key, value] for key, value in results.items()], headers=[
              'metrics', 'result']), '\n')
        
        for metric, value in results.items():
            results_all[model_name][metric].append(value)

    
import pandas as pd
df_results = pd.read_hdf('results_for_ausgrid5models.h5', key = 'results')
# Convert results_all to DataFrame
df_results = pd.concat({k: pd.DataFrame(v) for k, v in results_all.items()}, axis=0)

# Reset the index and rename the columns for easier plotting
df_results.reset_index(inplace=True)
df_results.rename(columns={'level_0': 'model', 'level_1': 'fold'}, inplace=True)

means = df_results.groupby(['model']).mean().drop(columns = ['fold'])
mins = df_results.groupby(['model']).min().drop(columns = ['fold'])
maxs = df_results.groupby(['model']).max().drop(columns = ['fold'])




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

order = np.argsort(means['ROC-AUC'])[::-1]

# assuming means, maxs, and mins are your DataFrame
metrics = means.columns
models = means.index[order]

x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8), dpi = 120)

# Create a bar for each metric
for i, metric in enumerate(metrics):
    mean = means[metric][order]
    min_val = mins[metric][order]
    max_val = maxs[metric][order]
    ax.bar(x + i * width, mean, width, yerr=[mean-min_val, max_val-mean], capsize=5, label=metric)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('The 5 models performance on synthetic attacks')
ax.set_xticks(x + width*(len(metrics)-1)/2)


ax.set_xticklabels(models)
ax.legend()
plt.show()



