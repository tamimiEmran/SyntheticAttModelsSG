

'''
This experiment will load all attacks. Hyper tune the model. Train the model and test them model.
Then the model will be tested on the real attacks


There will be 2 datasets each fold.

mainset: is the benign set with the synthetic attacks
    a. divided to three (train, validation, testing)
testset: the real attacks


'''
import exp3_abstraction_hypertuning as hp

import utils as u
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import exp3_abstraction_loading_data as load
from applyAttacks import synth_real_datasets
        
# constants for this experiments are:
attack_types = [*range(0, 13), 'ieee']
#%%
import matplotlib.pyplot as plt

# Set default figure size
plt.rcParams["figure.figsize"] = [10, 6]  # width, height in inches

# Set default dpi
plt.rcParams["figure.dpi"] = 100  # dpi value

# Set default yticks font size
plt.rcParams['ytick.labelsize'] = 16  # size in points
plt.rcParams['xtick.labelsize'] = 16  # size in points
plt.rcParams['axes.labelsize'] = 18  # size in points
plt.rcParams['axes.titlesize'] = 20  # size in points
    
# Set default padding for titles
plt.rcParams['axes.titlepad'] = 20  # pad in points

# Set default padding for x, y labels
plt.rcParams['axes.labelpad'] = 20  # pad in points
    
 




def metrics_func(model, set_):
    x, y = set_
    
    predictions = model.predict_proba(x)
    metrics_dicts = metrics.classification_report(y, predictions[:, 1] > 0.5 , output_dict=True)
    fpr, tpr, thresholds_testset = metrics.roc_curve(y, predictions[:, 1])
    metrics_dicts['AUC'] = metrics.auc(fpr, tpr)
    
    
    
    return metrics_dicts


def average_AUC(metrics_list):
    average_auc = 0
    
    for lst in metrics_list:
        average_auc = average_auc + lst['AUC']
        
    return average_auc/len(metrics_list)


metrics_train = []
metrics_validation = []
metrics_test = []

metrics_testset = []
attack_types = [*range(0, 13), 'ieee']




def avg_auc_(attack_types):
    metrics_testset = []
    metrics_test = []
    for foldId in range(1, 11):
    
        train, validation, test = load.load_attack_data(foldId, 'synthetic', attack_types)
        
        params_class = hp.models(validation, 'synthetic')    
        parameters = params_class.parameters_of('catboost')
        
        
        testset = load.load_attack_data(foldId, 'real', attack_types)
        
        testset = np.vstack([tuple[0] for tuple in testset]), np.hstack([tuple[1] for tuple in testset])
        
    
    

        # parameters = {'depth': 9, 'learning_rate': 0.5847618988751239}
        model = CatBoostClassifier(**parameters, verbose= False)
        model.fit(*train)
        
        # metrics_train.append(metrics_func(model, train))
        # metrics_validation.append(metrics_func(model, validation))
        metrics_test.append(metrics_func(model, test))
    
        metrics_testset.append(metrics_func(model, testset))
    
        # print(f'{foldId}: the auc for validation set', metrics_validation[-1]['AUC'])
        print(f'{foldId}: the auc for testing set', metrics_test[-1]['AUC'])
    
        print(f'{foldId}: the auc on the real attacks', metrics_testset[-1]['AUC'])
        
        
    return average_AUC(metrics_testset), (metrics_testset,metrics_test)







def find_best_attack_removal(attack_types, avg_auc_func = avg_auc_, rtn_all = False):
    best_auc = 0
    best_attack_type = None



    if len(attack_types) == 1:
        return (0, attack_types[0]), avg_auc_func(attack_types)
    
    
    if rtn_all:
        return (None, 'All attacks'), avg_auc_func(attack_types)
    
    
    
    for i, attack_type in enumerate(attack_types):
        print(f"Removing attack type: {attack_type}")

        # Create a list without the current attack type
        remaining_attack_types = attack_types[:i] + attack_types[i+1:]
        print('remaining attack types', remaining_attack_types)
        # Compute AUC for the remaining attack types
        current_auc, temp_metrics = avg_auc_func(remaining_attack_types)
        
        # print('all except', attack_type ,'auc',current_auc)

        # Update best AUC and best attack type if necessary
        if current_auc > best_auc:
            best_auc = current_auc
            best_attack_type = attack_type
            best_metrics = temp_metrics
            index_of_att = i

    return (index_of_att, best_attack_type), (best_auc,best_metrics)


def plot_avg_errorbars(all_attacks_auc = None, attackes_removed = None):
    if all_attacks_auc == None:
        all_attacks_auc = extract_attacks_auc_per_fold()[:-1]
    if attackes_removed == None:
        attackes_removed = np.load('new_atts_removed_cont.npy')[:-1]
        
    
    
    


    # Replace this with your list of lists
    data = all_attacks_auc
    # Calculate the mean and standard deviation of each inner list
    means = [np.mean(inner_list) for inner_list in data]
    std_devs = [np.std(inner_list) for inner_list in data]
    
    # Set up the plot
    x_values = range(1, len(data) + 1)
    plt.errorbar(x_values, means, yerr=std_devs, fmt='o', capsize=5, capthick=2)
    
    # Customize the plot
    plt.title('The AUC for the ten folds')
    plt.xlabel('Starting with all attacks \n and iteratively removing an attack')
    plt.ylabel('AUC')
    plt.xticks(x_values, attackes_removed )
    
    # Show the plot
    plt.show()
    
    return 
    

def extract_attacks_auc_per_fold(iterative_best_metrics = None):
    if iterative_best_metrics is None:
        iterative_best_metrics = np.load('new_exp2_folds_cont.npy', allow_pickle= True)
    
    all_attacks_auc = []
    for att_all_metrics in iterative_best_metrics:
        att_all_metrics_real = att_all_metrics[0]
        attack_auc = []
        for fold in range(10):
            temp_auc = att_all_metrics_real[fold]['AUC']
            attack_auc.append(temp_auc)
        all_attacks_auc.append(attack_auc)
    
    return all_attacks_auc
    


def main(attack_types = [*range(0, 13), 'ieee']):
    
    
    
    iterative_best_metrics = list(np.load('new_exp2_folds_cont.npy', allow_pickle = True))
    attackes_removed = list(np.load('new_atts_removed_cont.npy'))
    
    while len(attack_types):
        
        best_attack_type, best_auc = find_best_attack_removal(attack_types, avg_auc_)
        iterative_best_metrics.append(best_auc[1])
        
        print(f"Best attack type to remove: {best_attack_type[1]}, AUC: {best_auc[0]}")
        index_of_att = best_attack_type[0]
        
        np.save('new_exp2_folds_cont.npy', iterative_best_metrics)
        
        removed = attack_types.pop(index_of_att)
        
        attackes_removed.append(removed)
        np.save('new_atts_removed_cont.npy', attackes_removed)
    
    
    plot_avg_errorbars( extract_attacks_auc_per_fold(iterative_best_metrics), attackes_removed)



















