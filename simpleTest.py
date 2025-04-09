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
from joblib import dump
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, TimeDistributed
import matplotlib.pyplot as plt
import Loader as loader


def compute_metrics(y_test, predictions):
    # Apply a threshold to convert predictions into binary labels
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)
    
    # Calculate the metrics
    f1 = f1_score(y_test, binary_predictions)
    roc_auc = roc_auc_score(y_test, predictions)
    precision = precision_score(y_test, binary_predictions)
    recall = recall_score(y_test, binary_predictions)
    
    return {"F1 score": f1, "ROC-AUC": roc_auc, "Precision": precision, "Recall": recall}

def reshapeX(x):
    time_steps = 7
    features = x.shape[1] // time_steps
    x = x[:,:time_steps*features]
    print(x.shape)
    x_list = []
    for example in x:
        temp_x = example.reshape(time_steps, features )
        x_list.append(np.expand_dims(temp_x, axis = 0))
    
    examples = np.concatenate(x_list, axis = 0)
    
    return examples

        

def label_stats(array):
    array = array.flatten()
    total_samples = len(array)
    unique_labels, counts = np.unique(array, return_counts=True)
    label_percentages = counts / total_samples * 100

    stats = {
        "total_samples": total_samples,
        "labels": {}
    }
    
    for label, count, percentage in zip(unique_labels, counts, label_percentages):
        stats["labels"][label] = {
            "count": count,
            "percentage": percentage
        }

    return stats

class CNNLSTM:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=self.input_shape)

        # CNN layers
        conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding = 'same')(input_layer)
        pool1 = MaxPooling1D(pool_size=2, padding = 'same')(conv1)
        conv2 = Conv1D(filters=64, kernel_size=3, activation='relu')(pool1)
        pool2 = MaxPooling1D(pool_size=2)(conv2)

        # LSTM layer
        lstm = LSTM(128)(pool2)

        # Fully connected layers
        dense1 = Dense(64, activation='relu')(lstm)
        dropout1 = Dropout(0.5)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.5)(dense2)
        output_layer = Dense(1, activation='sigmoid')(dropout2)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

        return model

    def fit(self, x, y, batch_size=32, epochs=10, validation_split=0.15):
        history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
        plt.figure()
        plt.plot(np.arange(1, len(history.history['loss']) + 1), history.history['loss'], label='Training Loss')
        plt.plot(np.arange(1, len(history.history['val_loss']) + 1), history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
        
    def predict(self, x):
        return self.model.predict(x)

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

#load data
data_class = loader.Loader()
data_class.realVsSynth_Monthlyexamples(foldID=1)
x, y = data_class.real_train_examples, data_class.real_train_labels

tst = data_class.real_test_examples, data_class.real_test_labels
# y = y.values
#%%
print('tr', label_stats(y))
# print('tst', label_stats(tst[1].values))
print('tst', label_stats(tst[1]))

#fit model
model = SVC()
# modelCat = CatBoostClassifier(silent=True)


#original data shape (66501, 1035). (consumers, days)
# time_steps = 7 # one week
# features = x.shape[1] // time_steps
# x_cnn = reshapeX(x)

# input_shape = (time_steps, features)
# model_cnnLSTM = CNNLSTM(input_shape)
# model_cnnLSTM.fit(x_cnn, y, batch_size= 32 , epochs= 90)
x_test, y_test = tst
# y_test = y_test.values
model.fit(x, y)
#%%
# for examples_to_take in range(int(len(x)*0.1), len(x), len(x)//10):
#     modelCat.fit(x[:examples_to_take],y[:examples_to_take])
#     predictionsCat = modelCat.predict_proba(x_test)
    
#     print(f'\n catboost with training percentage of {examples_to_take/len(x)} \n', roc_auc_score(y_test, predictionsCat[:,1]))
#     print('\ncatboost\n',evaluate_model(modelCat, x_test, y_test))

#evaluate model

predictions = model.decision_function(x_test)

# x_test_cnn = reshapeX(x_test)
# predictions_cnnLstm = model_cnnLSTM.predict(x_test_cnn)
roc_auc = roc_auc_score(y_test, predictions)
print("SVC ROC-AUC:\n", roc_auc)
# print('\ncat\n', roc_auc_score(y_test, predictionsCat[:,1]))

# print('\ncatboost\n',evaluate_model(modelCat, x_test, y_test))
print('\nsvm\n',evaluate_model(model, x_test, y_test))
# print('\ncnn-lstm\n',compute_metrics(y_test, predictions_cnnLstm))
