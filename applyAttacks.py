import sys
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.insert(1, r'D:\Initial_workspace')
import attackTypes
import random
import matplotlib.pyplot as plt
from attackTypes import changeMonth
from tqdm import tqdm
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import iqr
from scipy.stats import mstats
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.decomposition import PCA

def plot_sample_data(sample_original_data, sample_modified_data, sample_consumer, sample_month):
    plt.figure(figsize=(12, 6))
    plt.plot(sample_original_data, label='Original Data', marker='o')
    plt.plot(sample_modified_data, label='Modified Data', marker='o')
    plt.title(f'Sample Consumer: {sample_consumer} in Month: {sample_month}')
    plt.xlabel('Days')
    plt.ylabel('Consumption')
    plt.legend()
    plt.show()


def attackDF(dataframe, attack_type):
    print('attack df of shape', dataframe.shape, 'att:', attack_type)
    
    if 'Year-Month' in dataframe.columns:
        dataframe = dataframe.drop('Year-Month', axis=1)
    
    monthGroups = dataframe.groupby(dataframe.index.to_period('M'))
    
    modified_groups = []

    sample_consumer = random.choice(dataframe.columns.unique())
    
    for group_name, group_data in tqdm(monthGroups):
        modified_group = changeMonth(attack_type, group_data)
        modified_groups.append(modified_group)
        
        sample_consumer = random.choice(dataframe.columns.unique())

        sample_original_group = group_data
        sample_modified_group = modified_group
    
    sample_original_data = sample_original_group[sample_consumer]
    sample_modified_data = sample_modified_group[sample_consumer]
    plot_sample_data(sample_original_data, sample_modified_data, sample_consumer, group_name)

    result_df = pd.concat(modified_groups)
    # Check if input and output DataFrames have the same columns
    assert set(dataframe.columns) == set(result_df.columns), "Columns in input and output DataFrames do not match."
    
    # Check if the number of rows in the input and output DataFrames are the same
    assert len(dataframe.index) == len(result_df.index), "Number of rows in input and output DataFrames do not match."
    result_df.index = dataframe.index
    return result_df





def pad_and_stats(npArr):
    monthPerExample = 1
    
    means = npArr.mean(axis=1).reshape(-1, 1)
    std = npArr.std(axis=1).reshape(-1, 1)
    mins = npArr.min(axis=1).reshape(-1, 1)
    maxs = npArr.max(axis=1).reshape(-1, 1)
    # skewness = np.apply_along_axis(mstats.skew, 1, npArr).reshape(-1, 1)
    # kurt = np.apply_along_axis(mstats.kurtosis, 1, npArr).reshape(-1, 1)
    # medians = np.median(npArr, axis=1).reshape(-1, 1)
    # iqr_values = np.apply_along_axis(iqr, 1, npArr).reshape(-1, 1)
    # range_values = (maxs - mins)
    # energy = np.sum(npArr**2, axis=1).reshape(-1, 1)

    
    '''
    npArr = np.hstack((means, std, mins, maxs, medians, skewness, kurt, iqr_values, range_values, energy, npArr))
    numOfFeatures = 10
    '''
    
    npArr = np.hstack((means, std, mins, maxs, npArr))
    numOfFeatures = 4
    npArr = np.pad(npArr, ((
        0, 0), (0, 31 * monthPerExample + numOfFeatures  - npArr.shape[1])), constant_values=(0.0))
    
    return npArr







def examplesFromDF(dataframe):
    if 'Year-Month' in dataframe.columns:
        dataframe = dataframe.drop('Year-Month', axis=1)
    
    # Group the data by consumer and month
    grouped = dataframe.groupby(dataframe.index.to_period('M'))

    # Initialize an empty list to store valid examples
    examples = []

    for group_name, group_data in grouped:
        # Create a 2D numpy array for each group, with one row for each day
        example = group_data.values  # shape is (feature, consumer)
        example = example.T  # shape is (consumer, feature)
        
        # Check if there are more than 4 zero-readings in the example
        # zero_readings = np.sum(example == 0, axis=1)  # Count the zero-readings along each consumer
        # valid_consumers = zero_readings <= 4  # the consumers with 4 or fewer zero readings

        # example = example[valid_consumers]

        # Apply the pad_and_stats function to add crafted features to the example
        example_with_features = pad_and_stats(example)
        
        # Append example_with_features to examples list if not empty
        if example_with_features.size > 0:
            examples.append(example_with_features)

    # Combine all valid examples into a single 2D numpy array
    examples_np = np.vstack(examples)

    return examples_np



import os

attack_types = [*range(0, 13), 'ieee']

def save_dataframe(attack_type, dataframe, big_dataframe=False):
    """
    1. Check if the dataframe is saved locally as f'df_att{attack_type}.h5'
    2. If not available, use attackDF(dataframe, attack_type) function.
    3. Save the dataframe
    
    4. Check if attack_type is -1. Then save the dataframe as is and name it 'original'
    5. If attack_type is not in [*range(0,13), 'ieee'] or -1 raise an error.
    """

    valid_attack_types = list(range(0, 13)) + ['ieee', -1]
    if attack_type not in valid_attack_types:
        raise ValueError("Invalid attack_type. Must be in [*range(0, 13), 'ieee'] or -1")

    if attack_type == -1:
        file_name = 'original.h5'
    else:
        file_name = f'df_att{attack_type}.h5'
    
    if not os.path.exists(file_name):
        if big_dataframe:
            parts = []
            num_columns = len(dataframe.columns)
            part_size = num_columns // 100
            num_parts = 100 if num_columns % 100 == 0 else 101
            
            if attack_type != -1:
                
                for i in range(num_parts):
                    start_idx = i * part_size
                    end_idx = (i + 1) * part_size if i != num_parts - 1 else num_columns
                    part = dataframe.iloc[:, start_idx:end_idx]
                    part = attackDF(part, attack_type)
                    # part.to_hdf(f'{file_name[:-3]}_part{i + 1}.h5', key='df', mode='w')
                    parts.append(part)
                    
            
            # for i in range(100):
            #     part = pd.read_hdf(f'{file_name[:-3]}_part{i + 1}.h5', key='df')
            #     parts.append(part)
            #     os.remove(f'{file_name[:-3]}_part{i + 1}.h5')

            dataframe = pd.concat(parts, axis=1)
            print('attacked shape:', dataframe.shape)
        else:
            if attack_type != -1:
                dataframe = attackDF(dataframe, attack_type)

        dataframe.to_hdf(file_name, key='df', mode='w')

    print(f"Dataframe saved as {file_name}")

def load_dataframe(attack_type):
    valid_attack_types = list(range(0, 13)) + ['ieee', -1]
    if attack_type not in valid_attack_types:
        raise ValueError("Invalid attack_type. Must be in [*range(0, 13), 'ieee'] or -1")

    if attack_type == -1:
        file_name = 'original.h5'
    else:
        file_name = f'df_att{attack_type}.h5'
    flag = False
    if attack_type in [valid_attack_types[2], valid_attack_types[4], valid_attack_types[6], valid_attack_types[11], valid_attack_types[12]  ]:
        flag = True
        
    
    
    if not os.path.exists(file_name): 
        dataframe = pd.read_hdf('original.h5', key='df')
        save_dataframe(attack_type, dataframe, big_dataframe = flag)
        result_df = load_dataframe(attack_type)
        assert set(dataframe.drop('Year-Month', axis = 1).columns) == set(result_df.columns), "Columns in input and output DataFrames do not match."
        # Load the dataframe from the file
    # dataframe = pd.read_hdf(file_name, key='df')[np.load('common_columns.npy')]
    
    # this is for testing remove later and uncomment above
    dataframe = pd.read_hdf(file_name, key='df')
    
    return dataframe




labels_file_name = 'original_labels.h5'

import utils


def get_fold_data(foldId):
    
    # Load the utils.fold(id) function which returns a tuple of two lists
    synthetic_indices, real_indices = utils.fold(foldId)

    # Load the labels from the HDF5 file
    labels = pd.read_hdf(labels_file_name, key='df')
    positive_indices = labels[labels == 1].index


    # Split the dataframe into three parts
    real_positive_indices = positive_indices
    synthetic_indices = synthetic_indices
    real_negative_indices = real_indices

    return real_positive_indices, synthetic_indices, real_negative_indices



def smote_decorator(func):
    def wrapper(*args, **kwargs):
        # Call the original function
        train, validation, test = func(*args, **kwargs)

        # Apply SMOTE to train data
        print('oversampling')
        smote = SMOTE(random_state=42)
        train_examples, train_labels = train
        train_examples_resampled, train_labels_resampled = smote.fit_resample(train_examples, train_labels)
        
        # Apply SMOTE to validation data
        validation_examples, validation_labels = validation
        validation_examples_resampled, validation_labels_resampled = smote.fit_resample(validation_examples, validation_labels)

        # Return the resampled train and validation data along with the original test data
        return (train_examples_resampled, train_labels_resampled.reshape(-1,1)), (validation_examples_resampled, validation_labels_resampled.reshape(-1,1)), test

    return wrapper

def adasyn_decorator(func):
    def wrapper(*args, **kwargs):
        train, validation, test = func(*args, **kwargs)
        adasyn = ADASYN(random_state=42)
        train_examples, train_labels = train
        train_examples_resampled, train_labels_resampled = adasyn.fit_resample(train_examples, train_labels)

        validation_examples, validation_labels = validation
        validation_examples_resampled, validation_labels_resampled = adasyn.fit_resample(validation_examples, validation_labels)

        return (train_examples_resampled, train_labels_resampled.reshape(-1,1)), (validation_examples_resampled, validation_labels_resampled.reshape(-1,1)), test

    return wrapper



@adasyn_decorator
def fullDataExamples(foldId):
    def examplesOfSplit(data, labels):
        positive_data = [datum for datum, label in zip(data, labels) if label == 1]
        negative_data = [datum for datum, label in zip(data, labels) if label == 0]
        
        positive_examples = examplesFromDF(df[positive_data])
        positive_labels = [1] * len(positive_examples)
        negative_examples = examplesFromDF(df[negative_data])
        negative_labels = [0] * len(negative_examples)
        
        # Concatenate examples and labels
        examples = np.vstack((positive_examples,negative_examples))
        example_labels = np.hstack((np.array(positive_labels),np.array(negative_labels))).reshape(-1,1)
        
        
        
        return examples, example_labels

    real_positive_indices, synthetic_indices, real_negative_indices = get_fold_data(foldId)
    
    positives, negatives = real_positive_indices, synthetic_indices + real_negative_indices
    
    consumer_labels = [1]*len(positives) + [0] * len(negatives)
    
    data = list(positives) + negatives
    
    # Split the data into training (70%), validation (15%), and testing (15%) sets
    train_data, temp_data, train_labels, temp_labels = train_test_split(data, consumer_labels, test_size=0.3, random_state=42, stratify=consumer_labels)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    df = load_dataframe(-1)    

    train = examplesOfSplit(train_data, train_labels)
    validation = examplesOfSplit(val_data, val_labels)
    test = examplesOfSplit(test_data, test_labels)
    
    
    return train, validation, test
    
    
@adasyn_decorator
def PCA_fullDataExamples(foldId, pcaPercent = 0.9):
    print("preparing data")
    def examplesOfSplit(data, labels):
        positive_data = [datum for datum, label in zip(data, labels) if label == 1]
        negative_data = [datum for datum, label in zip(data, labels) if label == 0]
        
        positive_examples = principal_df[positive_data].values.T
        positive_labels = [1] * len(positive_examples)
        negative_examples = principal_df[negative_data].values.T
        negative_labels = [0] * len(negative_examples)
        
        # Concatenate examples and labels
        examples = np.vstack((positive_examples,negative_examples))
        example_labels = np.hstack((np.array(positive_labels),np.array(negative_labels))).reshape(-1,1)
        
        
        
        return examples, example_labels

    real_positive_indices, synthetic_indices, real_negative_indices = get_fold_data(foldId)
    
    positives, negatives = real_positive_indices, synthetic_indices + real_negative_indices
    
    consumer_labels = [1]*len(positives) + [0] * len(negatives)
    
    data = list(positives) + negatives
    
    # Split the data into training (70%), validation (15%), and testing (15%) sets
    train_data, temp_data, train_labels, temp_labels = train_test_split(data, consumer_labels, test_size=0.3, random_state=42, stratify=consumer_labels)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    df = load_dataframe(-1)
    
    if pcaPercent > 0.99:
        principal_df = df
        
    else:
        principal_df = pca_df(df, pcaPercent)
    
    
    train = examplesOfSplit(train_data, train_labels)
    validation = examplesOfSplit(val_data, val_labels)
    test = examplesOfSplit(test_data, test_labels)
    
    
    return train, validation, test



def pca_df(df, pcaPercent = 0.9):
    print('PCA df')
    df_ = df.drop('Year-Month', axis=1)
    
    pca = PCA()
    principal_components = pca.fit_transform(df_.T.values)
    
    desired_variance = pcaPercent  # Adjust this value according to your requirements
    explained_variance_ratio = pca.explained_variance_ratio_
    n_components = np.argmax(np.cumsum(explained_variance_ratio) >= desired_variance) + 1
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_.T.values)
    principal_df = pd.DataFrame(data=principal_components.T, columns = df_.columns )
    
    return principal_df
    
    
    
    
def indecis_attacks(index, N):
    # Split the second part into N equal parts
    second_part_split = np.array_split(index, N)

    return second_part_split

def check_intersection(x_0, real_x_0, tolerance=1e-6):
    intersection_found = False
    for i, row_x_0 in tqdm(enumerate(x_0), desc = 'checking intersection'):
        for j, row_real_x_0 in enumerate(real_x_0):
            if np.allclose(row_x_0, row_real_x_0, atol=tolerance):
                print(f"Intersection found at index {i} in x_0 and index {j} in real_x_0")
                print(x_0[i], '\n',real_x_0[j])
                intersection_found = True
                if intersection_found:
                    return intersection_found
                
    if not intersection_found:
        print("No intersection found between x_0 and real_x_0")
        
    return intersection_found

def synth_real_datasets(foldId, attack_types):
    def filter_examples_with_excessive_zeros(examples, max_zeros, tolerance=1e-5):
        zero_counts = np.sum(np.isclose(examples, 0.0, atol=tolerance), axis=1)
        return examples[zero_counts <= max_zeros]
    
    
    def attackExamples(synthP_split):
        att_indecis = np.array_split(synthP_split, len(attack_types))    
    
        synthPExamples = []
        for ix, attack_type in enumerate(attack_types):
    
            #load the data frame
            df_att =load_dataframe(attack_type)
            #select only the synthetic consumers
            temp_df = df_att[att_indecis[ix]]
            #convert it to a 2d numpy array
            temp_x_1 = examplesFromDF(temp_df)
    
                
            synthPExamples.append(temp_x_1)
        
        
        return np.vstack(synthPExamples)
        
   
    acceptableZeroReadingsPerExample = 10 #keep in mind the number of months
    # print('loading original and fetching indices')
    df = load_dataframe(-1)
    
    real, synth = tr_val_tst_DF_indices(foldId)
    realP, realN = real
    synthP, synthN = synth
    
    tolerance = 1e-6
    
    # print('counting zeros for synthP')
    synthP_tr = synthP[0]
    ex_synthP_tr = examplesFromDF(df[synthP_tr])
    synthP_tr_zeros  = np.sum(np.isclose(ex_synthP_tr, 0.0, atol=tolerance), axis=1)
    
    synthP_val = synthP[1]
    ex_synthP_val = examplesFromDF(df[synthP_val])
    synthP_val_zeros  = np.sum(np.isclose(ex_synthP_val, 0.0, atol=tolerance), axis=1)

    synthP_tst = synthP[2]
    ex_synthP_tst = examplesFromDF(df[synthP_tst])
    synthP_tst_zeros  = np.sum(np.isclose(ex_synthP_tst, 0.0, atol=tolerance), axis=1)

    
    
    
 
    # print('creating examples')
    examples_real_positives_training = examplesFromDF(df[realP[0]])
    examples_real_positives_validation = examplesFromDF(df[realP[1]])
    examples_real_positives_testing = examplesFromDF(df[realP[2]])
    
    examples_real_negatives_training = examplesFromDF(df[realN[0]])
    examples_real_negatives_validation = examplesFromDF(df[realN[1]])
    examples_real_negatives_testing = examplesFromDF(df[realN[2]])
    
    examples_synth_nagatives_training = examplesFromDF(df[synthN[0]])
    examples_synth_nagatives_validation = examplesFromDF(df[synthN[1]])
    examples_synth_nagatives_testing = examplesFromDF(df[synthN[2]])
    
    examples_synth_positives_training = attackExamples(synthP[0])
    examples_synth_positives_validation = attackExamples(synthP[1])
    examples_synth_positives_testing = attackExamples(synthP[2])
    
    # print('filtering excessive zeros')
    examples_real_positives_training = filter_examples_with_excessive_zeros(examples_real_positives_training, acceptableZeroReadingsPerExample)
    examples_real_positives_validation = filter_examples_with_excessive_zeros(examples_real_positives_validation, acceptableZeroReadingsPerExample)
    examples_real_positives_testing = filter_examples_with_excessive_zeros(examples_real_positives_testing, acceptableZeroReadingsPerExample)
    
    examples_real_negatives_training = filter_examples_with_excessive_zeros(examples_real_negatives_training, acceptableZeroReadingsPerExample)
    examples_real_negatives_validation = filter_examples_with_excessive_zeros(examples_real_negatives_validation, acceptableZeroReadingsPerExample)
    examples_real_negatives_testing = filter_examples_with_excessive_zeros(examples_real_negatives_testing, acceptableZeroReadingsPerExample)
    
    examples_synth_nagatives_training = filter_examples_with_excessive_zeros(examples_synth_nagatives_training, acceptableZeroReadingsPerExample)
    examples_synth_nagatives_validation = filter_examples_with_excessive_zeros(examples_synth_nagatives_validation, acceptableZeroReadingsPerExample)
    examples_synth_nagatives_testing = filter_examples_with_excessive_zeros(examples_synth_nagatives_testing, acceptableZeroReadingsPerExample)
    
    
    examples_synth_positives_training = examples_synth_positives_training[synthP_tr_zeros <= acceptableZeroReadingsPerExample]
    examples_synth_positives_validation = examples_synth_positives_validation[synthP_val_zeros <= acceptableZeroReadingsPerExample]
    examples_synth_positives_testing = examples_synth_positives_testing[synthP_tst_zeros <= acceptableZeroReadingsPerExample]

  
    # print('stacking positives and negatives')
    real_training_X = np.vstack((examples_real_positives_training, examples_real_negatives_training))
    real_training_Y = np.concatenate((np.ones(len(examples_real_positives_training)), np.zeros(len(examples_real_negatives_training))))
    
    #do the same for the rest (real validation, real testing, synth training, synth validation, synth testing)
    
    real_validation_X = np.vstack((examples_real_positives_validation, examples_real_negatives_validation))
    real_validation_Y = np.concatenate((np.ones(len(examples_real_positives_validation)), np.zeros(len(examples_real_negatives_validation))))
    
    real_testing_X = np.vstack((examples_real_positives_testing, examples_real_negatives_testing))
    real_testing_Y = np.concatenate((np.ones(len(examples_real_positives_testing)), np.zeros(len(examples_real_negatives_testing))))
    
    synth_training_X = np.vstack((examples_synth_positives_training, examples_synth_nagatives_training))
    synth_training_Y = np.concatenate((np.ones(len(examples_synth_positives_training)), np.zeros(len(examples_synth_nagatives_training))))
    
    synth_validation_X = np.vstack((examples_synth_positives_validation, examples_synth_nagatives_validation))
    synth_validation_Y = np.concatenate((np.ones(len(examples_synth_positives_validation)), np.zeros(len(examples_synth_nagatives_validation))))
    
    synth_testing_X = np.vstack((examples_synth_positives_testing, examples_synth_nagatives_testing))
    synth_testing_Y = np.concatenate((np.ones(len(examples_synth_positives_testing)), np.zeros(len(examples_synth_nagatives_testing))))

    

    real_train_tuple = real_training_X, real_training_Y
    real_validation_tuple = real_validation_X, real_validation_Y
    real_testing_tuple = real_testing_X, real_testing_Y
    
    synth_training_tuple = synth_training_X, synth_training_Y
    synth_validation_tuple = synth_validation_X, synth_validation_Y
    synth_testing_tuple = synth_testing_X, synth_testing_Y
    
    
    real_dataset = real_train_tuple, real_validation_tuple, real_testing_tuple
    synth_dataset = synth_training_tuple, synth_validation_tuple, synth_testing_tuple
    
    
    return synth_dataset, real_dataset






def tr_val_tst_DF_indices(foldId):
    
    """
    The function returns the following in order 
        real = (train_pos_indices, val_pos_indices, test_pos_indices ), (train_neg_indices, val_neg_indices, test_neg_indices)
        
        synth = (train_unmod_indices, val_unmod_indices, test_unmod_indices), (train_mod_indices, val_mod_indices, test_mod_indices)

        return real, synth
    """
    
    def separate_positive_negative_indices(indices, labels):
        positive_indices = [index for index, label in zip(indices, labels) if label == 1]
        negative_indices = [index for index, label in zip(indices, labels) if label == 0]
        return positive_indices, negative_indices

    



    real_positive_indices, synthetic_indices, real_negative_indices = get_fold_data(foldId)
    real_positive_indices = list(real_positive_indices)
    
    
    half_len = len(synthetic_indices) // 2
    
    unmodified_indecis, modified_indecis = synthetic_indices[:half_len], synthetic_indices[half_len:]
        # Combine real_positive_indices and real_negative_indices
    combined_indices_real = real_positive_indices + real_negative_indices
    
    # Create a list of labels: 1 for real_positive_indices, 0 for real_negative_indices
    labels_real = [1] * len(real_positive_indices) + [0] * len(real_negative_indices)
    
    # Split the combined_indices list with stratified sampling
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        combined_indices_real, labels_real, test_size=0.3, random_state=42, stratify=labels_real)
    
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    train_pos_indices, train_neg_indices = separate_positive_negative_indices(train_indices, train_labels)
    val_pos_indices, val_neg_indices = separate_positive_negative_indices(val_indices, val_labels)
    test_pos_indices, test_neg_indices = separate_positive_negative_indices(test_indices, test_labels)


    
    combined_indices_synthetic = unmodified_indecis + modified_indecis


    labels_synthetic = [0] * len(unmodified_indecis) + [1] * len(modified_indecis)

        # Split the combined_indices_synthetic list with stratified sampling
    train_indices_synthetic, temp_indices_synthetic, train_labels_synthetic, temp_labels_synthetic = train_test_split(
        combined_indices_synthetic, labels_synthetic, test_size=0.3, random_state=42, stratify=labels_synthetic)
    
    val_indices_synthetic, test_indices_synthetic, val_labels_synthetic, test_labels_synthetic = train_test_split(
        temp_indices_synthetic, temp_labels_synthetic, test_size=0.5, random_state=42, stratify=temp_labels_synthetic)
    
    train_unmod_indices, train_mod_indices = separate_positive_negative_indices(train_indices_synthetic, train_labels_synthetic)
    val_unmod_indices, val_mod_indices = separate_positive_negative_indices(val_indices_synthetic, val_labels_synthetic)
    test_unmod_indices, test_mod_indices = separate_positive_negative_indices(test_indices_synthetic, test_labels_synthetic)
    
    
    real = (train_pos_indices, val_pos_indices, test_pos_indices ), (train_neg_indices, val_neg_indices, test_neg_indices)
    
    synth = (train_mod_indices, val_mod_indices, test_mod_indices) , (train_unmod_indices, val_unmod_indices, test_unmod_indices)
    check_for_intersection(real, synth)
    return real, synth

def common_columns(original_df, df_list):
    common_cols = set(original_df.columns)

    for df in df_list:
        common_cols.intersection_update(df.columns)

    return common_cols
def check_columns_equal(df1, df2):
    return set(df1.columns) == set(df2.columns)


    
def check_for_intersection(real, synth):
    train_pos_indices, val_pos_indices, test_pos_indices = real[0]
    train_neg_indices, val_neg_indices, test_neg_indices = real[1]
    train_mod_indices, val_mod_indices, test_mod_indices = synth[0]
    train_unmod_indices, val_unmod_indices, test_unmod_indices = synth[1]

    all_indices = [
        train_pos_indices, val_pos_indices, test_pos_indices,
        train_neg_indices, val_neg_indices, test_neg_indices,
        train_mod_indices, val_mod_indices, test_mod_indices,
        train_unmod_indices, val_unmod_indices, test_unmod_indices
    ]

    for i, current_indices in enumerate(all_indices):
        for j, other_indices in enumerate(all_indices):
            if i != j:
                intersection = np.intersect1d(current_indices, other_indices)
                assert len(intersection) == 0, f"Intersection found between index sets {i} and {j}"

    






def visualize(original_df, df_list):
    attack_types = [*range(0, 13), 'ieee']
    n_columns = 10
    n_months = 10
    ncols = 4
    nrows = (len(df_list) + ncols - 1) // ncols
    columns = random.sample(list(original_df.columns), n_columns)
    months = random.sample(sorted(list(set(original_df.index.to_period('M')))), n_months)

    for col, month in zip(columns, months):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15 * ncols, 5 * nrows))
        axes = axes.flatten()

        original_month_data = original_df.loc[original_df.index.to_period('M') == month, col]

        for i, df in enumerate(df_list):
            att = i
            if i in [12,13]:
                att = i
                i = i+1
                
            month_data = df.loc[df.index.to_period('M') == month, col]
            axes[i].plot(month_data, label=f'DF of att: {attack_types[att]}')
            axes[i].plot(original_month_data, label='Original DataFrame', linestyle='--', linewidth=2)
            axes[i].set_title(f'Column: {col} - Month: {month}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(col)
            axes[i].legend()

        plt.tight_layout()
        plt.show()


# visualize(original_df, df_list)









