import dataset_setup
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_curve, auc, precision_score, recall_score, accuracy_score
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
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

ALL_ATTACKS = [*range(13), 'ieee']


def Find_best_attack_to_remove(attackTypes = [*range(13), 'ieee']):
    
    
    
    avg_auc = []
    best_attack_to_remove = None
    best_attack_to_remove_auc = -999
    
    for ix, att in tqdm(enumerate(attackTypes), desc = 'finding first attack'):
        attacktypes2 = attackTypes[:ix] + attackTypes[ix + 1:]
        
        dataclass = dataset_setup.two_datasets(attacktypes2, ALL_ATTACKS)
        
        fold_auc = []
        for fold in range(3):
            data1, data2 = dataclass.data1and2(fold)

            
            model = CatBoostClassifier(silent=True)
            
            model.fit(*data1)
            
            results = evaluate_model(model, *data2)
            
            fold_auc.append(results['ROC-AUC'])
        
        current_auc = sum(fold_auc)/len(fold_auc)
        current_auc_std = st.stdev(fold_auc)
        avg_auc.append(current_auc)
        
        if current_auc > best_attack_to_remove_auc:
            best_attack_to_remove = att
            best_attack_to_remove_auc = current_auc
            best_attack_to_remove_auc_std = current_auc_std
            
    
    avg_auc_all = sum(avg_auc)/len(avg_auc)
    
    return best_attack_to_remove, (best_attack_to_remove_auc, best_attack_to_remove_auc_std)

def test_on_all_attacks(attackSet):
    
    attacks_auc = []
    
    for ix, att in enumerate(ALL_ATTACKS):
        dataclass = dataset_setup.two_datasets(attackSet, [att])
        
        fold_auc = []
        for fold in range(3):
            data1, data2 = dataclass.data1and2(fold)

            
            model = CatBoostClassifier(silent=True)
            
            model.fit(*data1)
            
            results = evaluate_model(model, *data2)
            
            fold_auc.append(results['ROC-AUC'])
        
        current_auc = sum(fold_auc)/len(fold_auc)
        
        attacks_auc.append(current_auc)
    
    return attacks_auc

#%%

initial_dataclass = dataset_setup.two_datasets(ALL_ATTACKS, ALL_ATTACKS)

fold_auc = []
for fold in range(3):
    data1, data2 = initial_dataclass.data1and2(fold)
    model = CatBoostClassifier(silent=True)
    model.fit(*data1)
    results = evaluate_model(model, *data2)
    fold_auc.append(results['ROC-AUC'])
current_auc = sum(fold_auc)/len(fold_auc)
current_auc_std = st.stdev(fold_auc)

# current_auc = 0.8884159285408806
starting_set = [*range(13), 'ieee']

optimal_set = ['Remove none']
optimal_set_auc = [current_auc]
optimal_set_auc_std = [current_auc_std]
#%%
while len(starting_set) - 1 :
    
    best_attack_to_remove, best_attack_to_remove_auc = Find_best_attack_to_remove(starting_set)
    optimal_set.append(best_attack_to_remove)
    optimal_set_auc.append(best_attack_to_remove_auc[0])
    optimal_set_auc_std.append(best_attack_to_remove_auc[1])
    starting_set.remove(best_attack_to_remove)
    
#%%
np.save('additiveFinal_set.npy', optimal_set)
np.save('additiveFinal_auc.npy', optimal_set_auc)
np.save('additiveFinal_auc_std.npy', optimal_set_auc_std)

#%%
optimal_set = np.load('additiveFinal_set.npy')
optimal_set_auc = np.load('additiveFinal_auc.npy')
optimal_set_auc_std = np.load('additiveFinal_auc_std.npy')


plt.figure(figsize=(16,9), dpi = 120)

x_label = ['Remove\nNone', *optimal_set[1:]]

x_label = [ str(i) if i != 'ieee' else '13'  for i in x_label]

yerr = optimal_set_auc_std

plt.plot(x_label, optimal_set_auc, linewidth = 3)
plt.errorbar(x_label, optimal_set_auc, yerr = yerr, color = 'blue')
plt.annotate('att. 7\nremaining\n', (x_label[-1], optimal_set_auc[-1]), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('Iteratively removing an attack to\nfind the optimal set', fontsize = 16)
plt.ylim(0.8, 1)
plt.xlabel('The set of attacks starting from all\n of them and removing one at a time', fontsize = 16)
plt.ylabel('The AUC of the optimal set on all the attacks', fontsize = 16)




    
    
    
    
    