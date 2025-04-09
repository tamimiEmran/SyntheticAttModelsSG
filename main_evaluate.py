import utils as u
import random
from dtaidistance.dtw import distance as dst
from sklearn.preprocessing import MinMaxScaler
import warnings
from math import isnan
from tqdm import tqdm


import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, r'D:\Initial_workspace')
import attackTypes
import helperFunctions as aid
tqdm.pandas()
warnings.filterwarnings('ignore')


def plotting_defaults():

    plt.rcParams.update({'font.size': 28})
    plt.rcParams["figure.dpi"] = 60
    plt.rcParams["figure.figsize"] = (16, 9)
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    plt.rc('legend', fontsize=16)


plotting_defaults()


def _experimentalSetup():
    d, theft = load_data()
    sna = (d.isna().sum() <= 103)
    d = d[sna[sna].index]
    theft = theft[d.columns[:-1]]
    dp = _preprocessTSG(d.drop('Year-Month', axis=1).apply(pd.to_numeric))
    dp['Year-Month'] = d['Year-Month']

    npDistances = np.empty(shape=(0, len(dp.columns) - 1))
    prev_idx = 0
    for idx, month in tqdm(dp.groupby(dp['Year-Month'])):
        if prev_idx == 0:
            prev_idx = idx
            continue

        pre_month = dp.groupby(dp['Year-Month']).get_group(prev_idx)

        distances = [dst(month[col], pre_month[col])
                     for col in month.columns[:-1]]
        npDistances = np.vstack((npDistances, np.array(distances)))


def load_data(directory=r"D:\evaluateAttacks\data.csv"):
    data = pd.read_csv(directory)

    thieves = data.T.loc['FLAG']

    data = data.T.iloc[2:]
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    index = pd.date_range(data.index[0], data.index[-1])
    data = data.reindex(index)
    data['Year-Month'] = pd.Series(data.index,
                                   index=data.index).apply(lambda x: aid.getYearMonth(str(x)))

    return data, thieves


def _preprocessTechniqueEnergies(dataframe):

    def _calculate(temp_values, mean):

        mask = temp_values >= mean
        return np.sum(temp_values[mask] * 0.1)

    def _preprocess(series):
        index = series.index
        values = series.values
        gmean = np.nanmean(values)

        def _update(ix):
            temp_values = values[ix-5: ix + 5]

            mean = np.nanmean(temp_values)
            factor = _calculate(temp_values, mean)

            if isnan(mean):
                mean = gmean
                temp_values[:] = mean
                factor = _calculate(temp_values, mean)

            return factor * mean

        for ix in range(5, len(index) - 4, 10):

            if isnan(series.iloc[ix]):

                series.iloc[ix] = _update(ix)

        series.ffill(inplace=True)

        return series

    return dataframe.apply(_preprocess)


###
def validConsumersPerMonth(dataframe):

    nullDF = dataframe.isnull().groupby(dataframe['Year-Month']).sum()
    takeDF = nullDF <= 4
    takeDF = takeDF.iloc[:, :-1]
    year_monthlyDF = {
        index: row[row].index for index, row in takeDF.iterrows()}

    return year_monthlyDF


def _preprocessTSG(dataframe):
    dataframe = dataframe.apply(pd.to_numeric)
    dataframe.interpolate(inplace=True, limit=1)
    dataframe.fillna(0, inplace=True)

    threasholds = dataframe.mean() + 2 * dataframe.std()

    def _update(series):
        name = series.name
        thr = threasholds[name]

        series[series >= thr] = thr

        return series

    dataframe = dataframe.apply(_update)
    scaler = MinMaxScaler().fit_transform(dataframe)

    dataframe = pd.DataFrame(
        scaler, columns=dataframe.columns, index=dataframe.index)
    if dataframe.isna().sum().sum() != 0:
        raise Exception('The dataframe after preprocessing has nans')
    return dataframe


preprocess = {'energies': _preprocessTechniqueEnergies,
              'transactions': _preprocessTSG,
              'GBTD': False}


def preprocessed_data(preprocessing='transactions'):
    data, thieves = load_data()

    preProcessed = preprocess[preprocessing](data.drop(['Year-Month'], axis=1))
    preProcessed['Year-Month'] = data['Year-Month']
    return preProcessed, thieves

def dataframes(preprocessing='transactions'):
    data, thieves = load_data()

    preProcessed = preprocess[preprocessing](data.drop(['Year-Month'], axis=1))
    preProcessed['Year-Month'] = data['Year-Month']

    maskThieves = thieves[thieves == 1].index
    maskBenign = thieves[thieves == 0].index
    benign = preProcessed.iloc[:, [*maskBenign, -1]]
    theft = preProcessed.iloc[:, [*maskThieves, -1]]

    return benign, theft


'''

def create_examples(dataframe, attackType = 1):
    
    year_monthlyDF = validConsumersPerMonth(dataframe.replace(0.0, np.nan))
    
    modified = un_modified = np.empty(shape = (0, 31 + 4 *  preprocess['GBTD'] ))
    # restModified = rest_unModified  = np.empty(shape = (0, 31))
    
    first_run = True
    
    
    for key, consumers in tqdm(dataframe.groupby(dataframe['Year-Month'])):
        takeCons = year_monthlyDF[key]
        filteredConsumers = consumers.loc[:, takeCons].copy()
        con = np.random.randint(0,filteredConsumers.shape[1])
        # filteredConsumers.iloc[:, con].plot(title = f'{key} attack type ORIGINAL for con: {con}' , color ='blue')
        # filteredConsumers = preprocess[preprocessing](filteredConsumers)
        # filteredConsumers.fillna( filteredConsumers.groupby(filteredConsumers.index.weekday).transform('mean'), inplace=True )
        # filteredConsumers.ffill(inplace=True)
        # filteredConsumers.fillna( filteredConsumers.groupby(filteredConsumers.index.weekday).transform('mean'), inplace=True )
        
        filteredConsumers_attacked = attackTypes.changeMonth(attackType= attackType, dataframe = filteredConsumers)
        
        def visual_check(df1 = filteredConsumers, df2 = filteredConsumers_attacked):
            
            
            ax = df1.iloc[:, con].plot(title = f'{key} attack type {attackType} for con: {con}' , color ='blue')
            df2.iloc[:,con].plot(ax = ax, color = 'red')
            plt.show()
        
        if first_run:
            visual_check()
            first_run = False
        
        if filteredConsumers_attacked.isnull().sum().sum() !=0:
            
            raise Exception('has nan in ')
            
    
        npArr = filteredConsumers_attacked.values.T
        if preprocess['GBTD']:
            means = npArr.mean(axis = 1).reshape(-1,1)
            std = npArr.std(axis = 1).reshape(-1,1)
            mins = npArr.min(axis = 1).reshape(-1,1)
            maxs = npArr.max(axis = 1).reshape(-1,1)
            
            npArr = np.hstack((means,std,mins,maxs, npArr))
            
        npArr = np.pad(npArr, ( (0, 0), ( 0, 31 + 4 *  preprocess['GBTD']  - npArr.shape[1])), constant_values = ( 0.0))

        npArrOriginal = filteredConsumers.values.T
        if preprocess['GBTD']:
            means = npArrOriginal.mean(axis = 1).reshape(-1,1)
            std = npArrOriginal.std(axis = 1).reshape(-1,1)
            mins = npArrOriginal.min(axis = 1).reshape(-1,1)
            maxs = npArrOriginal.max(axis = 1).reshape(-1,1)
            
            npArrOriginal = np.hstack((means,std,mins,maxs, npArrOriginal))
        
        npArrOriginal = np.pad(npArrOriginal, ( (0, 0), ( 0, 31 + 4 *  preprocess['GBTD']  - npArrOriginal.shape[1])), constant_values = ( 0.0))
        
        # sample npArr here
        # sample_indices = np.random.choice(npArr.shape[0], size= int(npArr.shape[0] * 0.5), replace=False)
        # rest_indices = np.setdiff1d(np.arange(npArr.shape[0]), sample_indices)
        
        modified = np.vstack((modified,npArr))
        un_modified = np.vstack((un_modified,npArrOriginal))
        # print('should be plotting')
        plt.title(f'attack Type: {attackType}, consumer {con}')
        plt.plot(modified[con])
        plt.plot(un_modified[con])
        
        plt.show()
        # restModified, rest_unModified = np.vstack((restModified,npArr[rest_indices])), np.vstack((rest_unModified,npArrOriginal[rest_indices]))
        
    
    
    #modified: the examples that are changed to a specific attack
    #un_modified: the original examples that got modified
    #restModified: the samples of modified attacks that were not taken into account
    #rest_unModified: the original examples that were not taken 
    #The process is; take 50 percent of the original dataset and modify it to represent attacks and the rest is to represent honest consumers

    
    return modified, un_modified
'''


def save_arrays(modified, un_modified, modified_path='modified.npy', un_modified_path='un_modified.npy', save_original=False):
    np.save(modified_path, modified)
    if save_original:
        np.save(un_modified_path, un_modified)


def load_arrays(attacktype, un_modified_path='original.npy', two_parts=False, load_original=True):
    if attacktype in ['ieee', 12, 11, 8, 6, 2]:
        two_parts = True

    if not two_parts:
        modified = np.load(f'modified_attack{attacktype}.npy')
    if two_parts:
        modified = np.vstack((np.load(f'modified_attack{attacktype}_part1.npy'), np.load(
            f'modified_attack{attacktype}_part2.npy')))

    if not load_original:
        return modified

    un_modified = np.load(un_modified_path)
    return modified, un_modified


def loadOriginalExamples():
    examples = load_arrays(0)[1]
    labels = np.array([0] * (len(examples))).reshape(-1, 1)

    return examples, labels


def split_index_randomly(index, n_parts):
    array = index

    np.random.shuffle(array)  # Shuffle the input array randomly
    np.random.shuffle(array)
    np.random.shuffle(array)

    # Split the array into two parts
    split_index = len(array) // 2
    first_part = array[:split_index]
    second_part = array[split_index:]

    # Split the second part into N equal partitions
    split_indices = np.linspace(0, len(second_part), n_parts + 1, dtype=int)
    second_part_partitions = [
        second_part[split_indices[i]:split_indices[i + 1]] for i in range(n_parts)]

    # Convert the partitions back to index objects
    first_part_index = pd.Index(first_part)
    second_part_partitions_indices = [
        pd.Index(part) for part in second_part_partitions]

    return first_part_index, second_part_partitions_indices


def create_dataset_for_attacks(attack_types):
    amount_of_attacks = len(attack_types)

    index = [*range(768501)]

    originalLabels = np.array([0] * 768501).reshape(-1, 1)
    theftLabels = np.array([1] * 768501).reshape(-1, 1)

    original_index, theft_indices = split_index_randomly(
        index, amount_of_attacks)

    original_examples = load_arrays(attack_types[0])[1][original_index]
    original_labels = originalLabels[original_index]

    original_examples = np.hstack((original_examples, original_labels))

    examples = [original_examples]
    for ix, attack in enumerate(attack_types):
        theft_examples = load_arrays(attack, load_original=False)[
            theft_indices[ix]]
        theft_labels = theftLabels[theft_indices[ix]]

        theft_examples = np.hstack((theft_examples, theft_labels))

        examples.append(theft_examples)

    dataset = np.vstack(examples)

    np.random.shuffle(dataset)
    np.random.shuffle(dataset)
    np.random.shuffle(dataset)

    dataset, labels = dataset[:, :-1], dataset[:, -1]

    return dataset, labels.astype(np.int32)


def create_dataset_for_1class(attack_types):
    amount_of_attacks = len(attack_types) + 1

    index = [*range(768501)]

    originalLabels = np.array([0] * 768501).reshape(-1, 1)
    theftLabels = np.array([1] * 768501).reshape(-1, 1)

    original_index, theft_indices = split_index_randomly(
        index, amount_of_attacks)

    original_examples = load_arrays(attack_types[0])[1][original_index]
    original_labels = originalLabels[original_index]

    original_examples = np.hstack((original_examples, original_labels))

    originalTest = load_arrays(attack_types[0])[1][theft_indices[0]]
    originalTestLabels = originalLabels[theft_indices[0]]

    originalTest = np.hstack((originalTest, originalTestLabels))

    examples = [originalTest]

    for ix, attack in enumerate(attack_types):
        theft_examples = load_arrays(attack, load_original=False)[
            theft_indices[ix + 1]]
        theft_labels = theftLabels[theft_indices[ix + 1]]

        theft_examples = np.hstack((theft_examples, theft_labels))

        examples.append(theft_examples)

    dataset = np.vstack(examples)

    np.random.shuffle(dataset)
    np.random.shuffle(dataset)
    np.random.shuffle(dataset)

    dataset, labels = dataset[:, :-1], dataset[:, -1]

    return original_examples[:, :-1], original_examples[:, -1].astype(np.int32), dataset, labels.astype(np.int32)


def load_theft_examples():
    return np.load('theft_examples_new.npy')


def create_dataset_realTheft():
    examples = load_theft_examples()
    labels = np.array([1] * (len(examples))).reshape(-1, 1)

    return examples, labels


def create_real_testingSet():

    real_theft_examples = np.hstack(create_dataset_realTheft())
    original_examples = np.hstack(loadOriginalExamples())

    dataset = np.vstack((real_theft_examples, original_examples))

    return dataset[:, :-1], dataset[:, -1].astype(np.int32)


def create_balanced_real_testingSet():
    real_theft_examples = np.hstack(create_dataset_realTheft())
    original_examples = np.hstack(loadOriginalExamples())
    np.random.shuffle(original_examples)
    original_examples = original_examples[:len(real_theft_examples)]

    dataset = np.vstack((real_theft_examples, original_examples))

    np.random.shuffle(dataset)
    np.random.shuffle(dataset)
    np.random.shuffle(dataset)

    return dataset[:, :-1], dataset[:, -1].astype(np.int32)


def create_examples(dataframe, attackType=1, preprocess={'GBTD': True}):

    def pad_and_stats(npArr, preprocess):
        if preprocess['GBTD']:
            means = npArr.mean(axis=1).reshape(-1, 1)
            std = npArr.std(axis=1).reshape(-1, 1)
            mins = npArr.min(axis=1).reshape(-1, 1)
            maxs = npArr.max(axis=1).reshape(-1, 1)
            npArr = np.hstack((means, std, mins, maxs, npArr))
        npArr = np.pad(npArr, ((
            0, 0), (0, 31 + 4 * preprocess['GBTD'] - npArr.shape[1])), constant_values=(0.0))
        return npArr

    year_monthlyDF = validConsumersPerMonth(dataframe.replace(0.0, np.nan))

    modified_list = []
    un_modified_list = []

    first_run = random.random() < 0.001

    for key, consumers in dataframe.groupby(dataframe['Year-Month']):
        takeCons = year_monthlyDF[key]

        if not len(takeCons):
            continue

        filteredConsumers = consumers.loc[:, takeCons].copy()
        con = np.random.randint(0, filteredConsumers.shape[1])

        filteredConsumers_attacked = attackTypes.changeMonth(
            attackType=attackType, dataframe=filteredConsumers)

        if filteredConsumers_attacked.isnull().sum().sum() != 0:
            raise Exception('has nan in ')

        npArr = pad_and_stats(filteredConsumers_attacked.values.T, preprocess)
        npArrOriginal = pad_and_stats(filteredConsumers.values.T, preprocess)

        if first_run:
            plt.title(f'attack Type: {attackType}, consumer {con}')
            plt.plot(filteredConsumers_attacked.iloc[:, con])
            plt.plot(filteredConsumers.iloc[:, con])

            plt.show()
            first_run = False

        modified_list.append(npArr)
        un_modified_list.append(npArrOriginal)

    if len(modified_list):
        modified = np.concatenate(modified_list, axis=0)
        un_modified = np.concatenate(un_modified_list, axis=0)
    else:
        modified = None
        un_modified = None

    # save_arrays(modified, un_modified, modified_path= f'modified_attack{attackType}.npy', un_modified_path= 'original.npy', save_original = False)

    return modified, un_modified


def create_list_of_a_single_attack(dataframe, attackType, add_original=False):
    modified_examples_list = []
    unmodified_examples_list = []

    for ix, consumer in tqdm(enumerate(dataframe.columns[:-1])):

        temp_df = u.select(dataframe, consumer)
        # create modified examples of a 2d consumer
        # create un-modified examples of a 2d consumer
        modified_examples, unmodified_examples = create_examples(
            temp_df, attackType=attackType)

        modified_examples_list.append(modified_examples)
        unmodified_examples_list.append(unmodified_examples)

    # append them to the folder
    u.add2dExamples(f'att{attackType}.pkl',
                    modified_examples_list, initial=True)

    # if the original is created then comment it out.
    if add_original:
        u.add2dExamples('original.pkl', unmodified_examples_list, initial=True)


def _attack12(dataframe, attackType=12, preprocess={'GBTD': True}):

    def pad_and_stats(npArr, preprocess):
        if preprocess['GBTD']:
            means = npArr.mean(axis=1).reshape(-1, 1)
            std = npArr.std(axis=1).reshape(-1, 1)
            mins = npArr.min(axis=1).reshape(-1, 1)
            maxs = npArr.max(axis=1).reshape(-1, 1)
            npArr = np.hstack((means, std, mins, maxs, npArr))
        npArr = np.pad(npArr, ((
            0, 0), (0, 31 + 4 * preprocess['GBTD'] - npArr.shape[1])), constant_values=(0.0))
        return npArr

    year_monthlyDF = validConsumersPerMonth(dataframe.replace(0.0, np.nan))

    modified_list = []
    un_modified_list = []

    first_run = random.random() < 0.001

    for key, consumers in tqdm(dataframe.groupby(dataframe['Year-Month'])):
        takeCons = year_monthlyDF[key]

        if not len(takeCons):
            continue

        filteredConsumers = consumers.loc[:, takeCons].copy()
        con = np.random.randint(0, filteredConsumers.shape[1])

        filteredConsumers_attacked = attackTypes.changeMonth(
            attackType=attackType, dataframe=filteredConsumers)

        if filteredConsumers_attacked.isnull().sum().sum() != 0:
            raise Exception('has nan in ')

        # npArr = pad_and_stats(filteredConsumers_attacked.values.T, preprocess)
        # npArrOriginal = pad_and_stats(filteredConsumers.values.T, preprocess)

        if first_run:
            plt.title(f'attack Type: {attackType}, consumer {con}')
            plt.plot(filteredConsumers_attacked.iloc[:, con])
            plt.plot(filteredConsumers.iloc[:, con])

            plt.show()
            first_run = False

        modified_list.append(filteredConsumers_attacked)
        # un_modified_list.append(npArrOriginal)

    # save_arrays(modified, un_modified, modified_path= f'modified_attack{attackType}.npy', un_modified_path= 'original.npy', save_original = False)

    return modified_list


def temp(list_att12, dataframe, preprocess={'GBTD': True}):
    def pad_and_stats(npArr, preprocess):
        if preprocess['GBTD']:
            means = npArr.mean(axis=1).reshape(-1, 1)
            std = npArr.std(axis=1).reshape(-1, 1)
            mins = npArr.min(axis=1).reshape(-1, 1)
            maxs = npArr.max(axis=1).reshape(-1, 1)
            npArr = np.hstack((means, std, mins, maxs, npArr))
        npArr = np.pad(npArr, ((
            0, 0), (0, 31 + 4 * preprocess['GBTD'] - npArr.shape[1])), constant_values=(0.0))
        return npArr

    # list_att12 = _attack12(dataframe)

    list_of_2darrays_for_consumer = []

    for consumer in dataframe.columns[:-1]:
        atleast_once = False
        consumerLst = []
        for df in list_att12:
            if consumer in df.columns:
                atleast_once = True
                consumerNpArr = pad_and_stats(
                    df[[consumer]].values.T, preprocess)
                consumerLst.append(consumerNpArr)

        if atleast_once:
            tot2DArr = np.concatenate(consumerLst, axis=0)
        else:
            tot2DArr = None

        list_of_2darrays_for_consumer.append(tot2DArr)

    u.add2dExamples('att12.pkl', list_of_2darrays_for_consumer, initial=True)


def create_examples_theft(dataframe, preprocess={'GBTD': True}):

    year_monthlyDF = validConsumersPerMonth(dataframe)

    original = np.empty(shape=(0, 31 + 4 * preprocess['GBTD']))

    first_run = True

    for key, consumers in tqdm(dataframe.groupby(dataframe['Year-Month'])):
        takeCons = year_monthlyDF[key]
        filteredConsumers = consumers.loc[:, takeCons].copy()
        # filteredConsumers.fillna( filteredConsumers.groupby(filteredConsumers.index.weekday).transform('mean'), inplace=True )
        # filteredConsumers.ffill(inplace=True)
        # filteredConsumers.fillna( filteredConsumers.groupby(filteredConsumers.index.weekday).transform('mean'), inplace=True )

        def visual_check(df1=filteredConsumers):

            df1.iloc[:, 10:12].plot(title=f'{key} attack type', color='blue')
            plt.show()

        if first_run:
            visual_check()
            first_run = False

        if filteredConsumers.isnull().sum().sum() != 0:

            raise Exception('has nan in ')

        npArrOriginal = filteredConsumers.values.T
        if preprocess['GBTD']:
            means = npArrOriginal.mean(axis=1).reshape(-1, 1)
            std = npArrOriginal.std(axis=1).reshape(-1, 1)
            mins = npArrOriginal.min(axis=1).reshape(-1, 1)
            maxs = npArrOriginal.max(axis=1).reshape(-1, 1)

            npArrOriginal = np.hstack((means, std, mins, maxs, npArrOriginal))

        npArrOriginal = np.pad(npArrOriginal, ((
            0, 0), (0, 31 + 4 * preprocess['GBTD'] - npArrOriginal.shape[1])), constant_values=(0.0))

        # sample npArr here

        original = np.vstack((original, npArrOriginal))

    np.save('theft_examples_new.npy', original)
    return original


def create_theft_consumers(preProcessed_dataframe, attackType=1):

    select_thives = np.random.choice(
        preProcessed_dataframe.columns[:-1], size=int(preProcessed_dataframe.shape[1] * 0.5), seed=1)
    thieves = preProcessed_dataframe[select_thives].copy()

    select_normal = np.setdiff1d(
        preProcessed_dataframe.columns[:-1], select_thives)
    normal = preProcessed_dataframe[select_normal].copy()

    for idx, monthDF in tqdm(thieves.groupby(preProcessed_dataframe['Year-Month'])):

        modifiedMonthDf = attackTypes.changeMonth(
            attackType=attackType, preProcessed_dataframe=monthDF)
        ax = monthDF.iloc[:, 0, 10].plot(color='blue')
        modifiedMonthDf.iloc[:, 0, 10].plot(color='red', ax=ax)
        plt.plot()

        thieves.update(modifiedMonthDf)

    return thieves, normal


if __name__ == '__main__':

    pass
