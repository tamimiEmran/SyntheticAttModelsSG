import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import arr_10fold
from imblearn.over_sampling import SMOTE, ADASYN
from applyAttacks import attackDF
from tqdm import tqdm
loc = r"D:\evaluateAttacks\data.csv"

def smote_decorator(func):
    def wrapper(*args, **kwargs):
        # Call the original function
        train, test = func(*args, **kwargs)

        # Apply SMOTE to train data
        smote = SMOTE(random_state=42)
        train_examples, train_labels = train
        train_examples_resampled, train_labels_resampled = smote.fit_resample(train_examples, train_labels)
        

        # Return the resampled train and validation data along with the original test data
        return (train_examples_resampled, train_labels_resampled), test

    return wrapper

def adasyn_decorator(func):
    def wrapper(*args, **kwargs):
        train, test = func(*args, **kwargs)
        adasyn = ADASYN(random_state=42)
        train_examples, train_labels = train
        train_examples_resampled, train_labels_resampled = adasyn.fit_resample(train_examples, train_labels)
        

        
        return (train_examples_resampled, train_labels_resampled), test

    return wrapper   


def injectAttacks(df, attacktypes):
    assert not df['is_thief'].sum(), 'contains real thief'
    
    
    dfCopy = df.copy().drop(columns = ['is_thief'])


    ### choose 50% random consumers
    
    honset_index, theif_index = train_test_split(df.index, test_size = 0.5)
    
    for attackID, theifix_perAttack in enumerate(np.array_split(theif_index, len(attacktypes))):
        attackedDF = pd.read_hdf(f'full_DF_att{attacktypes[attackID]}', key='df')
        theft_df = attackedDF.loc[theifix_perAttack]
        dfCopy.update(theft_df, overwrite = True)
    
    
    
    dfCopy['is_thief'] = 0
    
    dfCopy.loc[theif_index, 'is_thief'] = 1    
    df.update(dfCopy)
    return df
    
    

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


def examplesFromDF(dataframe, labels):
    if 'Year-Month' in dataframe.columns:
        dataframe = dataframe.drop('Year-Month', axis=1)
    
    # Group the data by consumer and month
    grouped = dataframe.groupby(dataframe.index.to_period('M'))

    # Initialize an empty list to store valid examples
    examples = []
    labels = labels.values.reshape(-1,1)
    for group_name, group_data in grouped:
        # Create a 2D numpy array for each group, with one row for each day
        example = group_data.values  # shape is (feature, consumer)
        example = example.T  # shape is (consumer, feature)
        example = np.hstack((example,labels ))
        # Check if there are more than 4 zero-readings in the example
        # zero_readings = np.sum(example == 0, axis=1)  # Count the zero-readings along each consumer
        # valid_consumers = zero_readings <= 4  # the consumers with 4 or fewer zero readings

        # example = example[valid_consumers]

        # Apply the pad_and_stats function to add crafted features to the example
        example_with_features = pad_and_stats(example)
        temp_labels = example_with_features[:,-1].reshape(-1,1)
        
        assert np.array_equal(labels, temp_labels), "labels and temp_labels are not equal"
        
        # Append example_with_features to examples list if not empty
        if example_with_features.size > 0:
            examples.append(example_with_features)

    # Combine all valid examples into a single 2D numpy array
    examples_np = np.vstack(examples)
    labels_np = examples_np[:, -1]
    examples_np = examples_np[:,:-1]
    
    return examples_np, labels_np

class Loader : 
    def __init__(self, loc = loc , na_th = 0.6) -> None:
        data = pd.read_csv(loc)
        # fill the na columns in data by the mean of the previous and next values with a max gap of 1 
        data = data.fillna(method='ffill', limit=1)
        # calcuate na percentage for each row
        data['na_percentage'] = data.isna().sum(axis=1)/data.shape[1]
        # remove rows with na_percentage > na_th
        data = data[data['na_percentage'] <= na_th]
        # remove na_percentage column
        data = data.drop(columns=['na_percentage'])
        # fill na values to 0 
        data = data.fillna(0)
        # clip the data for each raw by the mean of the row  +- 2*std using pandas 
        # this return nans ! 
        # mean = data.iloc[:, 2:].mean(axis=1)
        # std = data.iloc[:, 2:].std(axis=1)
        # data.iloc[:, 2:] = data.iloc[:, 2:].clip(upper = mean +2 * std , axis=0)
        
        data.index = data['CONS_NO']
        data.drop(columns = ['CONS_NO'], inplace = True)
        
        
        
        thief_labels = data['FLAG']
        data.drop(columns = ['FLAG'], inplace = True)
        
        honest_consumers = list(thief_labels[thief_labels == 0].index)
        # self.folds = arr_10fold(honest_consumers, available = False)
        
        self.folds = np.load('10folds.npy', allow_pickle=True)
        
        self.thief_labels = thief_labels
        
        data.columns = pd.to_datetime(data.columns)
        
        data = data.sort_index(axis=1)
        
        self.data = data
                

    def load_data(self) :
        
        return self.data, self.thief_labels
    
    
    
    def realVSsynth_dataframe(self,foldID = 'From 1 to 10', loc = loc):
        assert isinstance(foldID, int), 'must be int from 1 to 10'
        
        #real dataframe for a fold
        data, thief_labels = self.load_data()
        
        theft_consumer = list(thief_labels[thief_labels == 1].index)

        real_set_forTheFold = theft_consumer + self.folds[foldID - 1][1] #The set of honset consumers to create a balance real dataframe
        
        ##synthetic dataframe for a fold
        synth_set_forTheFold = self.folds[foldID - 1][0] #The set of consumers to evaluate synthetic attacks
        
        
        realSet_df = data.loc[real_set_forTheFold]
        synthSet_df = data.loc[synth_set_forTheFold]
        
        realSet_df['is_thief'] = realSet_df.index.map(thief_labels)
        synthSet_df['is_thief'] = synthSet_df.index.map(thief_labels)

        assert not realSet_df['is_thief'].isna().sum() and not synthSet_df['is_thief'].isna().sum() ##make sure theres no na, that is no con ID is new
        assert len(set(realSet_df.index)) + len(set(synthSet_df.index)) == len(set(thief_labels.index)) ##make sure that the index are split correctly
        assert not len(set(realSet_df.index).intersection(set(synthSet_df.index))) ##make sure no consumer is in both sets
        assert not synthSet_df['is_thief'].sum() ##make sure no thief is in the synthetic df
    
        
    ###########
    
        return realSet_df, synthSet_df
    """
    def realVsSynth_split(self, foldID, test_size = 0.333):
        
        realSet_df, synthSet_df = self.realVSsynth_dataframe(foldID)

        # For realSet_df
        realSet_train, realSet_test = train_test_split(realSet_df, test_size=test_size, random_state=42)
        
        # For synthSet_df
        synthSet_train, synthSet_test = train_test_split(synthSet_df, test_size=test_size, random_state=42)
        
        return realSet_train, realSet_test, synthSet_train, synthSet_test
    """
    def realVsSynth_full(self, foldID, test_size= 0.333, oversample = True, attacktypes = ['ieee', *range(1,13)]):
        realSet_df, synthSet_df = self.realVSsynth_dataframe(foldID)
        
        realSet_train, realSet_test = train_test_split(realSet_df, test_size=test_size, random_state=42)
        
        
        
        
        if attacktypes is not None:
            synthSet_df = injectAttacks(synthSet_df, attacktypes )
        
        
        synthSet_train, synthSet_test = train_test_split(synthSet_df, test_size=test_size, random_state=42)
        
        

        
        # Initialize a scaler
        scaler = StandardScaler()
        
        realSet_train_shape = realSet_train.drop('is_thief', axis=1).shape
        realSet_test_shape = realSet_test.drop('is_thief', axis=1).shape
        
        # Fit on the training features and transform both training and testing features
        realSet_train_features_scaled = scaler.fit_transform(realSet_train.drop('is_thief', axis=1).values.reshape(-1,1)).reshape(realSet_train_shape)
        realSet_test_features_scaled = scaler.transform(realSet_test.drop('is_thief', axis=1).values.reshape(-1,1)).reshape(realSet_test_shape)
        
        real_train = realSet_train_features_scaled, realSet_train['is_thief']
        if oversample:
            adasyn = ADASYN()
            real_train_x_oversampled, real_train_y_oversampled = adasyn.fit_resample(realSet_train_features_scaled, realSet_train['is_thief'].values)
            
            real_train = real_train_x_oversampled, real_train_y_oversampled
        
        real_test = realSet_test_features_scaled, realSet_test['is_thief']
        
        scaler = StandardScaler()
        
        synthSet_train_shape = synthSet_train.drop('is_thief', axis=1).shape
        synthSet_test_shape = synthSet_test.drop('is_thief', axis=1).shape
        
        synthSet_train_features_scaled = scaler.fit_transform(synthSet_train.drop('is_thief', axis=1).values.reshape(-1,1)).reshape(synthSet_train_shape)
        synthSet_test_features_scaled = scaler.transform(synthSet_test.drop('is_thief', axis=1).values.reshape(-1,1)).reshape(synthSet_test_shape)
        
        synth_train = synthSet_train_features_scaled, synthSet_train['is_thief']
        synth_test = synthSet_test_features_scaled, synthSet_test['is_thief']
        
        return real_train, real_test, synth_train, synth_test
    
    def realVsSynth_Monthlyexamples(self, foldID,test_size= 0.333, oversample = False, attacktypes  = ['ieee', *range(0,13)]):
        con_ID, timeStamp = self.data.index, self.data.columns
        
        
        
        real_train, real_test, synth_train, synth_test = self.realVsSynth_full(foldID, test_size, oversample = False, attacktypes = attacktypes)
        
        real_train_cons, real_test_cons, synth_train_cons,synth_test_cons = real_train[1].index, real_test[1].index, synth_train[1].index, synth_test[1].index
        real_train_df, real_test_df = real_train[0], real_test[0]
        synth_train_df, synth_test_df = synth_train[0], synth_test[0]
        
        real_train_df, real_test_df = pd.DataFrame(real_train_df.T, columns= real_train_cons, index = timeStamp), pd.DataFrame(real_test_df.T, columns= real_test_cons, index = timeStamp)
        synth_train_df, synth_test_df = pd.DataFrame(synth_train_df.T, columns= synth_train_cons, index = timeStamp), pd.DataFrame(synth_test_df.T, columns= synth_test_cons, index = timeStamp)
        
        
        
        self.real_train_examples, self.real_train_labels = examplesFromDF(real_train_df, real_train[1])
        self.real_test_examples, self.real_test_labels = examplesFromDF(real_test_df, real_test[1])
        self.synth_train_examples, self.synth_train_labels = examplesFromDF(synth_train_df, synth_train[1])
        self.synth_test_examples, self.synth_test_labels = examplesFromDF(synth_test_df, synth_test[1])
        
        if oversample:
            overSample = ADASYN()
            self.real_train_examples, self.real_train_labels = overSample.fit_resample(self.real_train_examples, self.real_train_labels)  
        
    
    
    @adasyn_decorator
    def fullDataExamples_scaled(self,test_size= 0.3 ):
        data, thief_labels = self.load_data()
        
        df_train, df_test, thief_labels_train, thief_labels_test = train_test_split(data, thief_labels ,test_size= test_size, random_state=42, stratify = thief_labels)
        
        # thief_labels_train.index = thief_labels.index[thief_labels_train.index]
        # df_train.index = thief_labels.index[thief_labels_train.index]
        
        # thief_labels_test.index = thief_labels.index[thief_labels_test.index]
        # df_test.index = thief_labels.index[thief_labels_test.index]
        
        # Initialize a scaler
        scaler = StandardScaler()

        # Fit the scaler on the training data (excluding the 'is_thief' label) and transform both training and testing data
        train_scaled = scaler.fit_transform(df_train.values.reshape(-1,1)).reshape(df_train.shape)
        test_scaled = scaler.transform(df_test.values.reshape(-1,1)).reshape(df_test.shape)
        
        
        return (train_scaled, thief_labels_train), (test_scaled,thief_labels_test )
        
    @adasyn_decorator
    def monthlyExamples_scaled(self, test_size= 0.3 ):
        data, thief_labels = self.load_data()
        df_train, df_test, thief_labels_train, thief_labels_test = train_test_split(data, thief_labels ,test_size= test_size, random_state=42, stratify = thief_labels)
        
        thief_labels_train.index = thief_labels.index[thief_labels_train.index]
        thief_labels_test.index = thief_labels.index[thief_labels_test.index]
        
        
        self.M_Ex_all_labels = thief_labels_train, thief_labels_test
        
        scaler = StandardScaler()

        # Fit the scaler on the training data (excluding the 'is_thief' label) and transform both training and testing data
        train_scaled = scaler.fit_transform(df_train.values.reshape(-1,1)).reshape(df_train.shape)
        test_scaled = scaler.transform(df_test.values.reshape(-1,1)).reshape(df_test.shape)
        
        con_ID, timeStamp = self.data.index, self.data.columns
        
        
        
        
        tr_x_df = pd.DataFrame(train_scaled.T, columns= thief_labels_train.index, index = timeStamp)
        tr_x_monthly, tr_y_monthly = examplesFromDF(tr_x_df, thief_labels_train)
        
        self.M_df_tr = tr_x_df
        
        tst_x_df = pd.DataFrame(test_scaled.T, columns= thief_labels_test.index, index = timeStamp)
        tst_x_monthly, tst_y_monthly = examplesFromDF(tst_x_df, thief_labels_test)
        
        self.M_df_tst = tst_x_df
        
        return (tr_x_monthly, tr_y_monthly), (tst_x_monthly, tst_y_monthly)
        
    
def save_dataframe(attacktype, dataclass):
    data = dataclass.data
    dfCopy = data.T.copy()
    
    
    
    
    parts = []
    num_columns = len(dfCopy.columns)
    part_size = num_columns // 100
    num_parts = 100 if num_columns % 100 == 0 else 101
    

    for i in tqdm(range(num_parts)):
        start_idx = i * part_size
        end_idx = (i + 1) * part_size if i != num_parts - 1 else num_columns
        part = dfCopy.iloc[:, start_idx:end_idx]
        part = attackDF(part, attacktype)
        parts.append(part)
            
    

    attackedDF = pd.concat(parts, axis=1)
    
    
    
    
    assert attackedDF.T.shape == data.shape, 'something is wrong when applying the attack'
    
    
    
    
    
    
    attackedDF.T.to_hdf(f'full_DF_att{attacktype}', key='df', mode='w')
   
