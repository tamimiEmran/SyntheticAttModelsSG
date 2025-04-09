import pandas as pd
import random
from catboost import CatBoostClassifier
from exp3_abstraction_results import evaluate_model
import numpy as np
from applyAttacks import tr_val_tst_DF_indices, load_dataframe
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import Loader as l
from sklearn.model_selection import train_test_split
import exp3_abstraction_hypertuning as hp
from tqdm import tqdm


def pad_and_stats(npArr):
    monthPerExample = 1
    tempArr = npArr[:, :-1]
    means = tempArr.mean(axis=1).reshape(-1, 1)
    std = tempArr.std(axis=1).reshape(-1, 1)
    mins = tempArr.min(axis=1).reshape(-1, 1)
    maxs = tempArr.max(axis=1).reshape(-1, 1)
    
    npArr = np.hstack((means, std, mins, maxs, npArr))

    numOfFeatures = 4
    total_columns = 32 * monthPerExample + numOfFeatures
    padding_columns = max(0, total_columns - npArr.shape[1])

    npArr = np.pad(npArr, ((0, 0), (padding_columns, 0)), constant_values=(0.0))

    return npArr

def compute_metrics(true_labels, pred_labels):
    # Compute metrics using scikit-learn functions
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)

    return {'F1 score': f1, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}


def process_monthly_data(df, month):
    monthly_data = df[df['Year-Month'] == month]
    return monthly_data


def create_model(trn, val):


    
    
    
    

    # hyperTune = hp.models(val, 'allDataMonthlyExs')
    # parameters = hyperTune.parameters_of('catboost')
    # model = CatBoostClassifier(**parameters, silent=True )

    model = CatBoostClassifier( silent=True )
    model.fit(*trn)
    
    print(evaluate_model(model, *trn))
    print(evaluate_model(model, tst_x_monthly, tst_y_monthly))
    
    
    return model

def generate_anomaly_dataframe(df, model):
    
    dfCopy = df.copy()
    
    consumers = dfCopy.columns  
    # dfCopy['Date'] = dfCopy.index
    dfCopy['Year-Month'] = dfCopy.index.to_period('M')
    unique_months = dfCopy['Year-Month'].unique()

    # Create a new dataframe to store the anomaly predictions
    anomaly_df = pd.DataFrame(columns=consumers, index=unique_months)

    for consumer in tqdm(consumers, desc = 'generating anomaly predictions'):
        consumer_data = dfCopy[[consumer, 'Year-Month']]

        for month in unique_months:
            month_data = process_monthly_data(consumer_data, month)
            anomaly_prediction = predict_anomaly(month_data[consumer], model)
            anomaly_df.at[month, consumer] = anomaly_prediction
    
    
    del(dfCopy)
    return anomaly_df

# Define your `predict_anomaly` function here
def predict_anomaly(series, model = None):
    dummyLabel = np.array([999]).reshape(1,-1)
    npArr = series.values.reshape(1,-1)
    npArr = np.hstack((npArr, dummyLabel))
    example = pad_and_stats(npArr)
    example = example[:,:-1]
    # print(example, example.shape)
    
    prediction = model.predict_proba(example)[:,1][0]
    
    # Your model logic to predict anomaly
    return prediction


def evaluate_anomalyDF(anomaly_df, labels ,threashold, months):
    
    actual_anomalous = labels.index[labels==1]
    anomaly = anomaly_df>=threashold
    anomaly = anomaly.sum(axis=0 )
    
    predicted_anomalous = ((anomaly>=months)[anomaly>=months]).index
    # predicted_benign = ((anomaly<months)[anomaly<months]).index
    
    # Construct the confusion matrix for the predicted indices of the models and the actual indices.
    true_labels = [1 if consumer in actual_anomalous else 0 for consumer in anomaly_df.columns]
    
    
    
    pred_labels = [1 if consumer in predicted_anomalous else 0 for consumer in anomaly_df.columns]
    
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    return true_labels, pred_labels
    
#%%  
anomaly_dfs_tst = []
anomaly_dfs_val = []

confMatrix_val = []
confMatrix_tst = []

data_class = l.Loader()
(tr_x_monthly, tr_y_monthly), (tst_x_monthly, tst_y_monthly) = data_class.monthlyExamples_scaled()
X_train, X_val, y_train, y_val = train_test_split(tr_x_monthly, tr_y_monthly, test_size=0.1, random_state=42, stratify=tr_y_monthly)

#%% 
model = create_model((tr_x_monthly, tr_y_monthly), (X_val, y_val))
#%% 
trainingLabels, testingLabels = data_class.M_Ex_all_labels
trainingDF = data_class.M_df_tr
testingDF = data_class.M_df_tst
#%% 
anomalyDF_tr = generate_anomaly_dataframe(trainingDF, model)
anomalyDF_tst = generate_anomaly_dataframe(testingDF, model)


#%% 




best_combination = None
best_metric_value = 0
average_metrics = {}

for threshold in tqdm(range(0, 105, 5)):
    thr = threshold/100
    for month_thr in range(0,35,1):
        averageConfMatrix_val = None
        total_metrics = {'F1 score': 0, 'Precision': 0, 'Recall': 0, 'Accuracy': 0}

        true_labels, pred_labels = evaluate_anomalyDF(anomalyDF_tr, trainingLabels, thr, month_thr)
        
        # print(confMatrix)
        metrics = compute_metrics(true_labels, pred_labels)
        # print(metrics['F1 score'])
        for metric in total_metrics:
            total_metrics[metric] += metrics[metric]
        
        
        
        average_metrics[(thr, month_thr)] = {metric: total_metrics[metric] for metric in total_metrics}
    
        chosen_metric = 'F1 score'  # Choose the metric you want to optimize
        current_metric_value = average_metrics[(thr, month_thr)][chosen_metric]
        
        
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_combination = (thr, month_thr)
        
        


print(f"Best combination: Threshold = {best_combination[0]}, Month threshold = {best_combination[1]}")
print(f"Average metrics for the best combination: {average_metrics[best_combination]}")
confidence_tr = anomalyDF_tr.mean(axis = 0)

labels_monthly_actual = [ [i] * 34  for i in trainingLabels.values]
labels_monthly_actual = [item for sublist in labels_monthly_actual for item in sublist]

labels_monthly_prediction = anomalyDF_tr.values.T.flatten()
print(f'auc for the training set_Per consumer: {roc_auc_score(trainingLabels.values, confidence_tr.values)}')
print(f'auc for the training set_Per month: {roc_auc_score(labels_monthly_actual, labels_monthly_prediction)}')

#%%  
total_metrics = {'F1 score': 0, 'Precision': 0, 'Recall': 0, 'Accuracy': 0}


threashold, months = best_combination
true_labels, pred_labels = evaluate_anomalyDF(anomalyDF_tst, testingLabels, threashold, months)

metrics = compute_metrics(true_labels, pred_labels)
for metric in total_metrics:
    total_metrics[metric] += metrics[metric]



print("Average metrics for the test set \n", total_metrics)


confidence_tst = anomalyDF_tst.mean(axis = 0)

labels_monthly_actual = [ [i] * 34  for i in testingLabels.values]
labels_monthly_actual = [item for sublist in labels_monthly_actual for item in sublist]

labels_monthly_prediction = anomalyDF_tst.values.T.flatten()
print(f'auc for the training set_Per consumer: {roc_auc_score(testingLabels.values, confidence_tst.values)}')
print(f'auc for the training set_Per month: {roc_auc_score(labels_monthly_actual, labels_monthly_prediction)}')
