import dataset_setup
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_curve, auc, precision_score, recall_score, accuracy_score
import pandas as pd
from tqdm import tqdm
import numpy as np

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
    
    if roc_auc < 0.4:
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'AUC accuracy {accuracy_score(y_test, np.where(y_pred >= 0.5, 1, 0))}\n positive Percentage in labels: {np.sum(y_test) / len(y_test) * 100}')
        plt.legend(loc="lower right")
        plt.show()
    
    
    return {"F1 score": f1, "ROC-AUC": roc_auc, "precision": precision, "recall": recall}


results_df = pd.DataFrame(columns=['trained on', 'tested on' , 'Fold', 'F1 score', 'ROC-AUC', 'precision', 'recall'])
aucResults = []

attackTypes = [*range(13),'ieee']

results = { key:[] for key in attackTypes}

for train in tqdm(range(len(attackTypes)), desc = 'training on attack'):
    attacktypes1 = attackTypes[train]
    
    testAuc = []
    
    for i in tqdm(range(len(attackTypes)), desc = 'testing on attack'):

        attacktypes2 = attackTypes[i]
        
        dataclass = dataset_setup.two_datasets([attacktypes1], [attacktypes2])
        folds_auc = []
        for fold in tqdm(range(3), desc = 'folds'):
            
            data1, data2 = dataclass.data1and2(fold)
                    
            model = CatBoostClassifier(silent=True)
            
            model.fit(*data1)
            
            results = evaluate_model(model, *data2)
            folds_auc.append(results["ROC-AUC"])
            # results = {"F1 score": np.random.uniform(), "ROC-AUC": np.random.uniform(), "precision": np.random.uniform(), "recall": np.random.uniform()}
            print('trn attack', attackTypes[train], 'test', attackTypes[i], 'fold', fold, 'results', results)
            # results are a dictionary of the following {"F1 score": f1, "ROC-AUC": roc_auc, "precision": precision, "recall": recall}
            
            results_df = pd.concat([results_df, pd.DataFrame([{
                'trained on': attackTypes[train] if attackTypes[train] != 'ieee' else 13,
                'tested on' :attackTypes[i] if attackTypes[i] != 'ieee' else 13,
                'Fold': fold,
                **results  # this will expand the dictionary to fill in the rest of the columns
            }], index=[0])], ignore_index=True)
        
        folds_auc = sum(folds_auc)/len(folds_auc)
        testAuc.append(folds_auc)
    
    aucResults.append(testAuc)

results_df = pd.read_hdf('trainingOnEach_testingOnEach', key = 'results')

xticks_labels = [*range(14), 'Avg. tes-\nting AUC']
yticks_labels = [*range(14), 'Avg. dete-\nction AUC']
ticks = [i + 0.5 for i in range(15)]
auc_results = results_df.groupby(["trained on",  "tested on"]).mean(numeric_only = True).copy()


heatmapData = auc_results['ROC-AUC'].values.reshape(14,14)

meansAxis1 = heatmapData.mean(axis = 1).reshape(-1,1)
meansAxis0 = np.hstack((heatmapData.mean(axis = 0), np.nan)).reshape(1,-1)

heatmapData = np.hstack((heatmapData, meansAxis1))
heatmapData = np.vstack((heatmapData, meansAxis0))

import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(16,9), dpi = 180)

# create heatmap, with a red-green colormap
sns.heatmap(heatmapData, cmap='RdYlGn', linewidths=1, annot=True, vmin=0.5, vmax=1)

plt.xticks(ticks = ticks, labels = xticks_labels, rotation=0)  # Rotate xticks if necessary
plt.yticks(ticks = ticks, labels = yticks_labels , rotation=0)  # Rotate yticks if necessary




# Add labels and title
plt.xlabel('Attacks detected', fontsize = 16)  # replace 'ColumnX' with your actual column name
plt.ylabel('Training on a single attack',fontsize = 16)  # replace 'ColumnY' with your actual column name
plt.title('AUC score when training on a\nsingle attack and testing on the rest\n', fontsize = 20)

plt.show()





