

'''
This experiment will load one attack at a time. Hyper tune the model. Train the model and test them model.
Then the model will be tested on the real attacks


There will be 2 datasets each fold.

mainset: is the benign set with the synthetic attacks
    a. divided to three (train, validation, testing)
testset: the real attacks


'''
import exp3_abstraction_hypertuning as hp

import applyAttacks as load
import exp3_abstraction_loading_data as loading

from catboost import CatBoostClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


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
attack_lists_real = []
attack_lists_synthetic = []

atts_auc_synth = []
atts_auc_real = []
#%%
for attack_type in [*range(0, 13), 'ieee']:
    folds_auc_synth = []
    folds_auc_real = []
    
    for foldId in range(1, 11):
        
        
        
        train, validation, test = loading.load_attack_data(foldId, 'synthetic', attack_types= [attack_type])
        
        testset = loading.load_attack_data(foldId, 'real', attack_types= [attack_type])
        
        testset = np.vstack([tuple[0] for tuple in testset]), np.hstack([tuple[1] for tuple in testset])
        
    
        params_class = hp.models(validation, 'synthetic')    
        parameters = params_class.parameters_of('catboost')
    
        model = CatBoostClassifier(**parameters, verbose= False)
        model.fit(*train)
        
        metrics_train.append(metrics_func(model, train))
        metrics_validation.append(metrics_func(model, validation))
        metrics_test.append(metrics_func(model, test))
    
        metrics_testset.append(metrics_func(model, testset))
    
        print(f'{foldId}: the auc for testing set', metrics_test[-1]['AUC'])
        folds_auc_synth.append(metrics_test[-1]['AUC'])
        
    
        print(f'{foldId}: the auc on the real attacks', metrics_testset[-1]['AUC'])
        folds_auc_real.append(metrics_testset[-1]['AUC'])
        
        
    atts_auc_synth.append(folds_auc_synth)
    atts_auc_real.append(folds_auc_real)
    
    attack_lists_real.append(metrics_testset)
    attack_lists_synthetic.append(metrics_test)
    
    print(f'attack {attack_type} real set average auc:', average_AUC(metrics_testset))
    


for ix, attack_type in enumerate([*range(0, 13), 'ieee']):
    print(f'attack {attack_type} real set average auc:', average_AUC(attack_lists_real[ix]))

#%%

def compute_auc_for_attacks(elements, attack_types, average_AUC):
    num_attacks = len(attack_types)
    num_folds = len(elements) // num_attacks

    for i, attack_type in enumerate(attack_types):
        fold_list = elements[i * num_folds:(i + 1) * num_folds]
        auc = average_AUC(fold_list)
        print(f"Attack: {attack_type}, Average AUC: {auc}")



def plot_avg_errorbars(all_attacks_auc = None, attackes = [*range(0, 13), 'ieee']):

        
    
    
    


    # Replace this with your list of lists
    data = all_attacks_auc
    # Calculate the mean and standard deviation of each inner list
    means = [np.mean(inner_list) for inner_list in data]
    std_devs = [np.std(inner_list) for inner_list in data]
    
    # Set up the plot
    x_values = range(1, len(data) + 1)
    plt.errorbar(x_values, means, yerr=std_devs, fmt='o', capsize=5, capthick=2)
    
    # Customize the plot
    plt.title('The AUC for the ten folds with deviation: \n trained on synthetic tested on real')
    plt.xlabel('Attack ID')
    plt.ylabel('AUC')
    plt.xticks(x_values, attackes )
    
    # Show the plot
    plt.show()
    
    
def plot_auc(results_real, results_synthetic):
    model_names = [*range(0, 14)]
    n_models = len(model_names)
    avg_auc_real = [np.mean(model) for model in results_real]
    std_auc_real = [np.std(model) for model in results_real]
    avg_auc_synthetic = [np.mean(model) for model in results_synthetic]
    std_auc_synthetic = [np.std(model) for model in results_synthetic]

    x = np.arange(n_models) * 2  # The x locations for the groups
    width = 0.35  # The width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, avg_auc_real, width, yerr=std_auc_real, label='Real', capsize=5)
    rects2 = ax.bar(x + width/2, avg_auc_synthetic, width, yerr=std_auc_synthetic, label='Synthetic', capsize=5)

    ax.set_ylabel('AUC')
    ax.set_xlabel('Attack types')
    ax.set_title('AUC of the synthetic testing set attacks\n and then testing the model on the real set')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    fig.tight_layout()
    plt.show()


