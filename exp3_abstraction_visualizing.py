import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['figure.figsize'] = [10, 6]  # width, height in inches
plt.rcParams['figure.dpi'] = 100  # Resolution in dots per inch




def aucs(results_real, results_synthetic):
    model_names = list(results_real.keys())
    n_models = len(model_names)
    
    auc_real = [results_real[model]['ROC-AUC'] for model in model_names]
    
    auc_synthetic = [results_synthetic[model]['ROC-AUC'] for model in model_names]
    
    return auc_real, auc_synthetic

def auc_stats(auc_real, auc_synthetic):
    
    import scipy.stats as stats

    # Assuming your actual data is stored in the variables result1, result2, ..., result15
    F, p = stats.f_oneway(auc_real)

    print("F statistic for real:", F)
    print("p-value:", p)
    
    F_synth, p_synth = stats.f_oneway(auc_synthetic)
    
    print("F statistic for synthetic:", F_synth)
    print("p-value:", p_synth)

'''
def plot_auc(results_real, results_synthetic, results_synth_onReal):
    model_names = list(results_real.keys())
    n_models = len(model_names)
    
    avg_auc_real = np.array([np.mean(results_real[model]['ROC-AUC']) for model in model_names])
    std_auc_real = np.array([np.std(results_real[model]['ROC-AUC']) for model in model_names])
    
    avg_auc_synthetic = np.array([np.mean(results_synthetic[model]['ROC-AUC']) for model in model_names])
    std_auc_synthetic = np.array([np.std(results_synthetic[model]['ROC-AUC']) for model in model_names])
    
    avg_auc_synthetic_onreal = np.array([np.mean(results_synth_onReal[model]['ROC-AUC']) for model in model_names])
    std_auc_synthetic_onreal = np.array([np.std(results_synth_onReal[model]['ROC-AUC']) for model in model_names])
    
    
    # Sort by performance on real set
    order = avg_auc_real.argsort()[::-1]
    
    
    avg_auc_real = avg_auc_real[order]
    std_auc_real = std_auc_real[order]
    
    avg_auc_synthetic = avg_auc_synthetic[order]
    order_synth = avg_auc_synthetic.argsort()[::-1] + 1 
    std_auc_synthetic = std_auc_synthetic[order]
    
    avg_auc_synthetic_onreal = avg_auc_synthetic_onreal[order]
    order_synth_onreal = avg_auc_synthetic_onreal.argsort()[::-1] + 1
    std_auc_synthetic_onreal = std_auc_synthetic_onreal[order]
    
    model_names = [model_names[i] for i in order]
    # model_names_synth = [model_names[i] for i in order_synth]
    
    x = np.arange(n_models)  # The x locations for the groups
    width = 0.2  # The width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, avg_auc_real, width, yerr=std_auc_real, label='Real', capsize=5)
    rects2 = ax.bar(x + width/2, avg_auc_synthetic, width, yerr=std_auc_synthetic, label='Synthetic', capsize=5)

    # Add the ranking inside the bars
    for i, rect in enumerate(rects1):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01, str(i+1),
                ha='center', va='bottom', color='black')
    for i, rect in enumerate(rects2):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height()+ 0.01, str(order_synth[i]),
                ha='center', va='bottom', color='black')

    ax.set_ylabel('AUC')
    ax.set_title('AUC by Model and Data Type')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    fig.tight_layout()
    plt.show()
'''
def plot_auc(results_real, results_synthetic, results_synth_onReal):
    model_names = list(results_real.keys())
    n_models = len(model_names)
    
    avg_auc_real = np.array([np.mean(results_real[model]['ROC-AUC']) for model in model_names])
    std_auc_real = np.array([np.std(results_real[model]['ROC-AUC']) for model in model_names])
    
    avg_auc_synthetic = np.array([np.mean(results_synthetic[model]['ROC-AUC']) for model in model_names])
    std_auc_synthetic = np.array([np.std(results_synthetic[model]['ROC-AUC']) for model in model_names])
    
    avg_auc_synthetic_onreal = np.array([np.mean(results_synth_onReal[model]['ROC-AUC']) for model in model_names])
    std_auc_synthetic_onreal = np.array([np.std(results_synth_onReal[model]['ROC-AUC']) for model in model_names])
    
    
    # Sort by performance on real set
    order = avg_auc_real.argsort()[::-1]
    
    avg_auc_real = avg_auc_real[order]
    std_auc_real = std_auc_real[order]
    
    avg_auc_synthetic = avg_auc_synthetic[order]
    order_synth = avg_auc_synthetic.argsort()[::-1] + 1 
    std_auc_synthetic = std_auc_synthetic[order]
    
    avg_auc_synthetic_onreal = avg_auc_synthetic_onreal[order]
    order_synth_onreal = len(avg_auc_synthetic_onreal) - avg_auc_synthetic_onreal.argsort().argsort()
    std_auc_synthetic_onreal = std_auc_synthetic_onreal[order]
    
    model_names = [model_names[i] for i in order]

    x = np.arange(n_models)  # The x locations for the groups
    width = 0.2  # The width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, avg_auc_real, width, yerr=std_auc_real, label='Real', capsize=5)
    rects2 = ax.bar(x, avg_auc_synthetic, width, yerr=std_auc_synthetic, label='Synthetic', capsize=5)
    rects3 = ax.bar(x + width, avg_auc_synthetic_onreal, width, yerr=std_auc_synthetic_onreal, label='trained on synthetic\ntested on real', capsize=5)

    # Add the ranking inside the bars
    for i, rect in enumerate(rects1):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01, str(i+1),
                ha='center', va='bottom', color='black')
    for i, rect in enumerate(rects2):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01, str(order_synth[i]),
                ha='center', va='bottom', color='black')
    for i, rect in enumerate(rects3):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01, str(order_synth_onreal[i]),
                ha='center', va='bottom', color='black')

    ax.set_ylabel('AUC')
    ax.set_title('AUC by Model and Data Type')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    fig.tight_layout()
    plt.show()


def plot_auc_(results_real, results_synthetic):
    model_names = list(results_real.keys())
    n_models = len(model_names)
    avg_auc_real = [np.mean(results_real[model]['ROC-AUC']) for model in model_names]
    std_auc_real = [np.std(results_real[model]['ROC-AUC']) for model in model_names]
    avg_auc_synthetic = [np.mean(results_synthetic[model]['ROC-AUC']) for model in model_names]
    std_auc_synthetic = [np.std(results_synthetic[model]['ROC-AUC']) for model in model_names]

    x = np.arange(n_models) * 2  # The x locations for the groups
    width = 0.35  # The width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, avg_auc_real, width, yerr=std_auc_real, label='Real', capsize=5)
    rects2 = ax.bar(x + width/2, avg_auc_synthetic, width, yerr=std_auc_synthetic, label='Synthetic', capsize=5)

    ax.set_ylabel('AUC')
    ax.set_title('AUC by Model and Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    fig.tight_layout()
    plt.show()

def plot_precision_recall(results, title = 'Precision-Recall Curve for (real OR synthetic) Data'):
    plt.figure(figsize=(10, 8))

    for model_name, metrics in results.items():
        precision = metrics['precision']
        recall = metrics['recall']
        avg_f1 = np.mean(metrics['F1 score'])
        plt.scatter(recall, precision, marker='o', label=f'{model_name} (Avg F1: {avg_f1:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_model_performance(results, title='Model Performance for (real OR synthetic) Data'):
    plt.figure(figsize=(12, 8))

    model_names = list(results.keys())
    precision_means = [np.mean(metrics['precision']) for metrics in results.values()]
    recall_means = [np.mean(metrics['recall']) for metrics in results.values()]
    f1_means = [np.mean(metrics['F1 score']) for metrics in results.values()]

    bar_width = 0.25
    bar_positions = np.arange(len(model_names))

    plt.bar(bar_positions - bar_width, precision_means, width=bar_width, label='Precision')
    plt.bar(bar_positions, recall_means, width=bar_width, label='Recall')
    plt.bar(bar_positions + bar_width, f1_means, width=bar_width, label='F1 score')

    plt.xticks(bar_positions, model_names)
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title(title)
    plt.legend()
    plt.grid(axis='y')
    plt.show()

    
import pickle
def effect_of_trainingSize(model = 0):
    

    file_names = ["results_real50.pkl", "results_real60.pkl", "results_real70.pkl", "results_real80.pkl", "results_real90.pkl", "results_real100.pkl"]
    results = []
    
    for file_name in file_names:
        with open(file_name, "rb") as f:
            results.append(pickle.load(f))
    
    # Calculate the mean and standard deviation for each performance metric
    metrics = ["F1 score", "ROC-AUC", "precision", "recall"]
    means = []
    stds = []
    
    for result in results:
        model_name = list(result.keys())[model]
        mean_values = []
        std_values = []
        for metric in metrics:
            mean_values.append(np.mean(result[model_name][metric]))
            std_values.append(np.std(result[model_name][metric]))
        means.append(mean_values)
        stds.append(std_values)
    
    # Create a bar plot with error bars for each metric
    x = np.arange(len(metrics))  # The label locations
    width = 0.15  # The width of the bars
    
    fig, ax = plt.subplots()
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.bar(x - width/2 + i * width, mean, width, yerr=std, label=file_names[i])
    ax.set_ylim(0.3, 0.7)
    ax.set_ylabel("Scores")
    ax.set_title("Model Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    plt.show()





"""


results_real = {
    'CatBoost': {
        'F1 score': [0.8, 0.85, 0.78, 0.81, 0.77, 0.84, 0.79, 0.82, 0.76, 0.83],
        'AUC': [0.9, 0.92, 0.89, 0.93, 0.91, 0.94, 0.88, 0.9, 0.87, 0.92],
        'precision': [0.7, 0.75, 0.73, 0.8, 0.72, 0.76, 0.71, 0.79, 0.74, 0.82],
        'recall': [0.6, 0.65, 0.68, 0.7, 0.62, 0.69, 0.61, 0.66, 0.64, 0.72]
    },
    'XGBoost': {
        'F1 score': [0.75, 0.8, 0.79, 0.76, 0.78, 0.74, 0.77, 0.72, 0.79, 0.75],
        'AUC': [0.88, 0.9, 0.92, 0.91, 0.89, 0.93, 0.87, 0.9, 0.89, 0.91],
        'precision': [0.68, 0.73, 0.7, 0.76, 0.69, 0.74, 0.71, 0.7, 0.72, 0.75],
        'recall': [0.58, 0.63, 0.67, 0.65, 0.6, 0.64, 0.62, 0.61, 0.66, 0.69]
    },
    'RF': {
        'F1 score': [0.77, 0.82, 0.8, 0.81, 0.78, 0.83, 0.79, 0.76, 0.8, 0.85],
        'AUC': [0.87, 0.89, 0.9, 0.91, 0.92, 0.93, 0.88, 0.9, 0.91, 0.94],
        'precision': [0.72, 0.77, 0.75, 0.8, 0.78, 0.79, 0.76, 0.71, 0.74, 0.81],
        'recall': [0.62, 0.67, 0.69, 0.7, 0.68, 0.72, 0.63, 0.64, 0.65, 0.73]
    },
    'SVM': {
        'F1 score': [0.74, 0.78, 0.75, 0.76, 0.77, 0.79, 0.72, 0.73, 0.77, 0.8],
        'AUC': [0.85, 0.87, 0.88, 0.9, 0.89, 0.91, 0.84, 0.86, 0.88, 0.92],
        'precision': [0.66, 0.71, 0.68, 0.74, 0.73, 0.78, 0.7, 0.69, 0.72, 0.76],
        'recall': [0.56, 0.61, 0.64, 0.63, 0.6, 0.68, 0.6, 0.58, 0.65, 0.71]
    },
    'KNN': {
        'F1 score': [0.7, 0.73, 0.75, 0.72, 0.74, 0.71, 0.69, 0.7, 0.76, 0.73],
        'AUC': [0.8, 0.82, 0.84, 0.86, 0.85, 0.87, 0.81, 0.83, 0.86, 0.89],
        'precision': [0.64, 0.67, 0.66, 0.68, 0.7, 0.72, 0.65, 0.66, 0.74, 0.71],
        'recall': [0.54, 0.59, 0.62, 0.6, 0.58, 0.66, 0.57, 0.55, 0.64, 0.69]
    }
}


results_synthetic  = {
    'CatBoost': {
        'F1 score': [0.8, 0.85, 0.78, 0.81, 0.77, 0.84, 0.79, 0.82, 0.76, 0.83],
        'AUC': [0.9, 0.92, 0.89, 0.93, 0.91, 0.94, 0.88, 0.9, 0.87, 0.92],
        'precision': [0.7, 0.75, 0.73, 0.8, 0.72, 0.76, 0.71, 0.79, 0.74, 0.82],
        'recall': [0.6, 0.65, 0.68, 0.7, 0.62, 0.69, 0.61, 0.66, 0.64, 0.72]
    },
    'XGBoost': {
        'F1 score': [0.75, 0.8, 0.79, 0.76, 0.78, 0.74, 0.77, 0.72, 0.79, 0.75],
        'AUC': [0.88, 0.9, 0.92, 0.91, 0.89, 0.93, 0.87, 0.9, 0.89, 0.91],
        'precision': [0.68, 0.73, 0.7, 0.76, 0.69, 0.74, 0.71, 0.7, 0.72, 0.75],
        'recall': [0.58, 0.63, 0.67, 0.65, 0.6, 0.64, 0.62, 0.61, 0.66, 0.69]
    },
    'RF': {
        'F1 score': [0.77, 0.82, 0.8, 0.81, 0.78, 0.83, 0.79, 0.76, 0.8, 0.85],
        'AUC': [0.87, 0.89, 0.9, 0.91, 0.92, 0.93, 0.88, 0.9, 0.91, 0.94],
        'precision': [0.72, 0.77, 0.75, 0.8, 0.78, 0.79, 0.76, 0.71, 0.74, 0.81],
        'recall': [0.62, 0.67, 0.69, 0.7, 0.68, 0.72, 0.63, 0.64, 0.65, 0.73]
    },
    'SVM': {
        'F1 score': [0.74, 0.78, 0.75, 0.76, 0.77, 0.79, 0.72, 0.73, 0.77, 0.8],
        'AUC': [0.85, 0.87, 0.88, 0.9, 0.89, 0.91, 0.84, 0.86, 0.88, 0.92],
        'precision': [0.66, 0.71, 0.68, 0.74, 0.73, 0.78, 0.7, 0.69, 0.72, 0.76],
        'recall': [0.56, 0.61, 0.64, 0.63, 0.6, 0.68, 0.6, 0.58, 0.65, 0.71]
    },
    'KNN': {
        'F1 score': [0.7, 0.73, 0.75, 0.72, 0.74, 0.71, 0.69, 0.7, 0.76, 0.73],
        'AUC': [0.8, 0.82, 0.84, 0.86, 0.85, 0.87, 0.81, 0.83, 0.86, 0.89],
        'precision': [0.64, 0.67, 0.66, 0.68, 0.7, 0.72, 0.65, 0.66, 0.74, 0.71],
        'recall': [0.54, 0.59, 0.62, 0.6, 0.58, 0.66, 0.57, 0.55, 0.64, 0.69]
    }
}

"""








