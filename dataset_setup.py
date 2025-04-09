import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def attacked_df(normal_cons, theft_cons, attackTypes = [*range(13), 'ieee']):
    
    
    theft_cons_perAtt = np.array_split(theft_cons, len(attackTypes))
    
    chunks = []
    
    for consumers, attack_type in zip([normal_cons, *theft_cons_perAtt], [ 'original' ,*attackTypes]):
        if attack_type == 'original':
            chunks.append(pd.read_hdf("ausgrid_attacked.h5", key = 'original')[consumers])
            
        else:
            chunks.append(pd.read_hdf("ausgrid_attacked.h5", key = f'attack{attack_type}')[consumers])
            
    
    final_df = pd.concat(chunks, axis=1)
    
    return final_df

### set up the training and testing dataset for all attacks.

def stats(np2d):
    
    shape = np2d.shape
    
    
    
    
    # Assuming x_train is your original feature matrix
    x_mean = np.mean(np2d, axis=1, keepdims=True)
    x_std = np.std(np2d, axis=1, keepdims=True)
    x_max = np.max(np2d, axis=1, keepdims=True)
    x_min = np.min(np2d, axis=1, keepdims=True)

    # Concatenate the new features to the original feature matrix
    x_extended = np.concatenate([np2d, x_mean, x_std, x_max, x_min], axis=1)
    
    return x_extended

def daily_examples(df, normal_cons, theft_cons):
    
    assert len(set(normal_cons).intersection(set(theft_cons))) == 0, "normal_cons and theft_cons are not mutually exclusive."

    consumers_id = df.columns
    assert len(set(consumers_id)) == len(consumers_id), "Consumer IDs are not unique."

    
    examples_consumers = df.values.T
    numberOfConsumers = examples_consumers.shape[0]
    assert examples_consumers.shape[1] % 48 == 0, "Number of time intervals is not a multiple of 48."

    daysPerCon = examples_consumers.shape[1]//48
    
    labels = np.array([0 if con in normal_cons else 1 for con in consumers_id])
    labels = np.repeat(labels, daysPerCon)
    
    examples_daily = examples_consumers.reshape( numberOfConsumers*daysPerCon, 48 )
    
    examples_daily = stats(examples_daily)
    
    
    shuffled = np.random.permutation([*range(labels.shape[0])])
    
    
    
    return examples_daily[shuffled], labels[shuffled]

class data_loader():
    def __init__(self, attackTypes = [*range(13), 'ieee']):
        cons = np.array([*range(1,301)])
        np.random.shuffle(cons)
        half = len(cons)//2
        
        self.normal_cons, self.theft_cons = cons[:half], cons[half:]
        
        
        self.df = attacked_df(self.normal_cons, self.theft_cons, attackTypes)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        self.folds_train = []
        self.folds_test = []
        
        np.random.shuffle(cons)
        for train_index, test_index in kf.split(cons):
            train_consumers = [cons[i] for i in train_index]
            test_consumers = [cons[i] for i in test_index]
            
            self.folds_train.append(train_consumers)
            self.folds_test.append(test_consumers)
            
        
        
    def train_test_dailyExamples(self, foldid):
        train_set = daily_examples(self.df[self.folds_train[foldid]], self.normal_cons, self.theft_cons)
        
        x, y = train_set
        
        shape = x.shape
        
        scaler = StandardScaler().fit(x.reshape(-1,1))
        
        x = scaler.transform(x.reshape(-1,1)).reshape(shape)
        train_set = x,y
        
        test_set = daily_examples(self.df[self.folds_test[foldid]], self.normal_cons, self.theft_cons)
        
        x_tst, y_tst = test_set
        shape_test = x_tst.shape
        
        x_tst = scaler.transform(x_tst.reshape(-1, 1)).reshape(shape_test)
        test_set = x_tst, y_tst
        
        
        return train_set, test_set



class two_datasets():
    
    def __init__(self, attackTypes1, attackTypes2):
        cons = np.array([*range(1,301)])
        np.random.shuffle(cons)
        half = len(cons)//2
        
        self.normal_cons, self.theft_cons = cons[:half], cons[half:]
        
        np.random.shuffle(cons)

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        self.folds_data1 = []
        self.folds_data2 = []


        for train_index, test_index in kf.split(cons):
            train_consumers = [cons[i] for i in train_index]
            test_consumers = [cons[i] for i in test_index]
            
            self.folds_data1.append(train_consumers)
            self.folds_data2.append(test_consumers)
        
        self.df1 = attacked_df(self.normal_cons, self.theft_cons, attackTypes1)
        self.df2 = attacked_df(self.normal_cons, self.theft_cons, attackTypes2)
        
        
    def data1and2(self, foldid):
        
        data1 = daily_examples(self.df1[self.folds_data1[foldid]], self.normal_cons, self.theft_cons)
        
        x1, y1 = data1
        
        scaler = StandardScaler().fit(x1.reshape(-1,1))
        x1_shape = x1.shape
        x1 = scaler.transform(x1.reshape(-1,1)).reshape(x1_shape)
        
        data1 = x1, y1
        
        data2 = daily_examples(self.df2[self.folds_data2[foldid]], self.normal_cons, self.theft_cons)
        
        x2, y2 = data2
        x2_shape = x2.shape
        
        x2 = scaler.transform(x2.reshape(-1,1)).reshape(x2_shape)
        
        data2 = x2, y2
        
        return data1, data2
    




