import dataset_setup
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_curve, auc, precision_score, recall_score
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import statistics as st

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




def best_first_attack(optimalSet = None, attackTypes = [*range(13), 'ieee']):
    
    avg_auc = []
    best_generalizable_attack = None
    best_generalizable_attack_auc = -999
    
    for ix, att in tqdm(enumerate(attackTypes), desc = 'finding first attack'):
        if optimalSet is None:
            attacktypes1 = [att]
        else: 
            attacktypes1 = optimalSet
            
            
        attacktypes2 = attackTypes[:ix] + attackTypes[ix + 1:]
        
        dataclass = dataset_setup.two_datasets(attacktypes1, attackTypes)
        
        fold_auc = []
        for fold in range(3):
            data1, data2 = dataclass.data1and2(fold)

            
            model = CatBoostClassifier(silent=True)
            
            model.fit(*data1)
            
            results = evaluate_model(model, *data2)
            
            fold_auc.append(results['ROC-AUC'])
        
        current_auc = sum(fold_auc)/len(fold_auc)
        best_generalizable_attack_auc_std = st.stdev(fold_auc)
        avg_auc.append(current_auc)
        
        if current_auc > best_generalizable_attack_auc:
            best_generalizable_attack = att
            best_generalizable_attack_auc = current_auc
            
    avg_auc_all = sum(avg_auc)/len(avg_auc)
    
    return best_generalizable_attack, (avg_auc, best_generalizable_attack_auc_std)


def _auc_for_attacks(attacks_train, attacks_test = [*range(13), 'ieee']):
    initial_dataclass = dataset_setup.two_datasets(attacks_train, attacks_test)

    fold_auc = []
    for fold in range(3):
        data1, data2 = initial_dataclass.data1and2(fold)
        model = CatBoostClassifier(silent=True)
        model.fit(*data1)
        results = evaluate_model(model, *data2)
        fold_auc.append(results['ROC-AUC'])
    current_auc = sum(fold_auc)/len(fold_auc)
    current_auc_std = st.stdev(fold_auc)
    
    return current_auc, current_auc_std


def best_attack_picked(optimalSet):
    remaining = list(set([*range(13), 'ieee']) - set(optimalSet))
    
    best_auc = 0
    best_attack_ = None
    
    for att in remaining:
        trainOn = [*optimalSet, att]
        print(trainOn)

        current_auc, current_auc_std = _auc_for_attacks(trainOn)
        
        if current_auc > best_auc:
            best_auc = current_auc
            best_auc_std_ = current_auc_std
            best_attack_ = att
        
        print(best_attack_, best_auc)
    return best_attack_, (best_auc, best_auc_std_)


def least_detected_attack(attackTypes1, attackTypes2):
    
    worst_auc = []
    leastDetectedAttack = None
    leastDetectedAttack_auc = 999
    
    for ix, att in tqdm(enumerate(attackTypes2), desc = 'finding least detected attack'):
        
        dataclass = dataset_setup.two_datasets(attackTypes1, [att])
        
        fold_auc = []
        for fold in range(3):
            data1, data2 = dataclass.data1and2(fold)

            model = CatBoostClassifier(silent=True)

            model.fit(*data1)

            results = evaluate_model(model, *data2)

            fold_auc.append(results['ROC-AUC'])

        current_auc = sum(fold_auc)/len(fold_auc)
        worst_auc.append(current_auc)
        
        if current_auc < leastDetectedAttack_auc:
            leastDetectedAttack = att
            leastDetectedAttack_auc = current_auc
                    
    avg_auc = sum(worst_auc)/len(worst_auc)
    
    return leastDetectedAttack, worst_auc




optimalSet = []
optimalSet_auc = []
optimalSet_auc_std = []

while len(optimalSet) < 14:
    best_attack_, auc_results = best_attack_picked(optimalSet)
    optimalSet.append(best_attack_)
    optimalSet_auc.append(auc_results[0])
    optimalSet_auc_std.append(auc_results[1])

optimalSet = [ str(i) if i != 'ieee' else '13' for i in optimalSet]
#%%
# for ix, sublist in enumerate(optimalSet_auc):
#     ##add np.nan in sublist in the index depending of optimalSet[]
'''

import seaborn as sns



heatmap_data = np.full((len(optimalSet) + 1, len(optimalSet)), np.nan)
heatmap_data[0] = optimalSet_auc[0]

# Fill in the AUC scores
for row in range(1, len(optimalSet)):
    counter = 0
    for col in range(len(optimalSet)):
        
        if col not in optimalSet[:row]:
            heatmap_data[row, col] = optimalSet_auc[row][counter]
            counter = counter + 1
        else:
            pass




y_axis_labels = [*optimalSet[::-1], 'Each attack\navg. on all'][::-1]
x_axis_labels = optimalSet[::-1]
x_axis_labels_top = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
x_axis_labels = optimalSet[::-1]
top_x_axis_labels = ['Att ' + str(label) + 'tested\non rest' for label in x_axis_labels_top]

 
plt.figure(figsize=(19, 6), dpi = 120)
ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=x_axis_labels_top, yticklabels=y_axis_labels)
ax.grid(alpha = 0.5)
plt.xlabel("Attack ID that is tested on")
plt.ylabel("The optimal set starting\nfrom the top and adding the\nworst attack iteratively")
plt.title("AUC score for the iterative optimal set\neach other attack\n")

# Add a secondary x-axis at the top
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(ax.get_xticks())
ax2.set_xticklabels(top_x_axis_labels)
ax2.set_xlabel('The top row represents training on this attack and testing on the rest \n')
plt.show()
'''
#%%
np.save('iterativeFinal_set.npy', optimalSet)
np.save('iterativeFinal_auc.npy', optimalSet_auc)
np.save('iterativeFinal_auc_stc.npy', optimalSet_auc_std)

#%%
optimalSet = np.load('iterativeFinal_set.npy')
optimalSet_auc = np.load('iterativeFinal_auc.npy')
optimalSet_auc_std = np.load('iterativeFinal_auc_stc.npy')


plt.figure(figsize=(16,9), dpi = 120)

x_label = optimalSet

yerr = optimalSet_auc_std

plt.plot(x_label, optimalSet_auc, linewidth = 3)
plt.errorbar(x_label, optimalSet_auc, yerr = yerr, color = 'blue')
plt.title('Iteratively adding an attack to\nfind the optimal set', fontsize = 16)
plt.ylim(0.8, 1)
plt.xlabel('The set of attacks by adding one at a time', fontsize = 16)
plt.ylabel('The AUC of the optimal set on all the attacks', fontsize = 16)







