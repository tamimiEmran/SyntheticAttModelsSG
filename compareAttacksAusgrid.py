import dataset_setup
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_curve, auc, precision_score, recall_score
import pandas as pd
from tqdm import tqdm
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


results_df = pd.DataFrame(columns=['AttackType', 'Fold', 'F1 score', 'ROC-AUC', 'precision', 'recall'])


attackTypes = [*range(13),'ieee']
for i in tqdm(range(len(attackTypes)), desc = 'attacks'):
    # attacktypes1 = attackTypes[:i] + attackTypes[i + 1:]
    attacktypes2 = attackTypes[i]
    
    dataclass = dataset_setup.two_datasets(attackTypes, [attacktypes2])
    
    for fold in tqdm(range(3), desc = 'folds'):
        
        data1, data2 = dataclass.data1and2(fold)
        
        print('train size', data1[1].size, 'percnt',  sum(data1[1].flatten())/data1[1].size, 'test size','percnt',  sum(data2[1].flatten())/data2[1].size , data2[1].size)
        
        model = CatBoostClassifier(silent=True)
        
        model.fit(*data1)
        
        results = evaluate_model(model, *data2)
        
        # results = {"F1 score": np.random.uniform(), "ROC-AUC": np.random.uniform(), "precision": np.random.uniform(), "recall": np.random.uniform()}
        print('testing', attackTypes[i], 'for fold', fold, '\nresults', results)
        # results are a dictionary of the following {"F1 score": f1, "ROC-AUC": roc_auc, "precision": precision, "recall": recall}
        
        results_df = pd.concat([results_df, pd.DataFrame([{
            'AttackType': attackTypes[i] if attackTypes[i] != 'ieee' else 13,
            'Fold': fold,
            **results  # this will expand the dictionary to fill in the rest of the columns
        }], index=[0])], ignore_index=True)
        



import seaborn as sns
import matplotlib.pyplot as plt

# Reshape DataFrame
df_melted = results_df.melt(id_vars=['AttackType', 'Fold'], var_name='Metric', value_name='Value')

# Create catplot
sns.catplot(data=df_melted, x='AttackType', y='Value', hue='Metric', kind='bar', errorbar='sd', height=6, aspect=2)

plt.title('Comparison of Metrics for Different Attack Types')
plt.ylabel('Score')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reshape DataFrame
df_melted = results_df.melt(id_vars=['AttackType', 'Fold'], var_name='Metric', value_name='Value')

metrics = df_melted['Metric'].unique()
attack_types = df_melted['AttackType'].unique()

x = np.arange(len(attack_types))  # label locations
width = 0.2  # width of the bars

fig, ax = plt.subplots(figsize = (16,9), dpi = 120)

# Generate a color dictionary for each metric
colors = ['r', 'g', 'b', 'y']
color_dict = dict(zip(metrics, colors))

# Plot each metric
# for i, metric in enumerate(metrics):
metric = 'ROC-AUC'
    
metric_data = df_melted[df_melted['Metric'] == metric]
means = metric_data.groupby('AttackType')['Value'].mean().values
mins = metric_data.groupby('AttackType')['Value'].min().values
maxs = metric_data.groupby('AttackType')['Value'].max().values

means_sorted_idx = np.argsort(means)[::-1]
means_sorted = means[means_sorted_idx]
attack_types_sorted = attack_types[means_sorted_idx]
# metrics_sorted = metrics[means_sorted_idx]

x_sorted = np.arange(len(attack_types_sorted))
errors = [np.array(means - mins)[means_sorted_idx], np.array(maxs - means)[means_sorted_idx]]
ax.bar(x_sorted - width/2 +  1*width/2, means_sorted, width, yerr=errors, label=metric, color=color_dict[metric])

  # Updated label locations
ax.set_xticks(x_sorted)
ax.set_xticklabels(np.hstack((attack_types_sorted[:-1], [13])), fontsize=22)  # Updated tick labels

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUC', fontsize = 22)
ax.set_xlabel('Attack tested on', fontsize = 22)
ax.set_title('Performance of the model when trained \non all attacks and tested on one', fontsize = 22)
ax.set_xticks(x)
attack_types_sorted = [i if i != 'ieee' else 13 for i in attack_types_sorted]
ax.set_xticklabels(attack_types_sorted, fontsize = 24)
# ax.set_yticklabels(fontsize = 18)
plt.yticks(fontsize=24)
plt.grid(True)
plt.locator_params(axis='y', nbins=10)
ax.legend(fontsize = 20)
fig.tight_layout()

plt.show()


