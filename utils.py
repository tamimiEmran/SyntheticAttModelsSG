import numpy as np
import pickle

def load_theft_examples():
    return np.load('theft_examples.npy')


def create_dataset_realTheft():
    examples = load_theft_examples()
    labels = np.array([1] * (len(examples))).reshape(-1, 1)

    return examples, labels


def arr_10fold(columns=None, file_name='10folds.npy', available=True):

    if available:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data

    '''
    

    Parameters
    ----------
    columns : a list of dataframe's honest columns
        The total number of columns.
    '10folds.npy' : string
        where to store the resulting array.

    Returns
    
    -------
    a list of tuples. The index of the list refers to the validation fold and the elements of the tuples are the columns for training and testing respectively
    The testing set is steched with the real thieves and the rest is the synthetic. 
    '''

    assert columns is not None

    # Calculate the number of columns in each fold
    num_columns = len(columns)
    # Create an array of indices representing the columns
    indices = np.arange(num_columns)

    # Randomly shuffle the indices
    np.random.shuffle(indices)

    # Split the indices into 10 equal parts (folds)
    folds = np.array_split(indices, 10)

    # Create a list to store the resulting tuples (training, testing)
    result = []

    # Iterate through each fold and create the training and testing sets
    for i in range(10):
        test_columns = folds[i]
        train_columns = np.concatenate([folds[j] for j in range(10) if j != i])

        # Convert the indices back to column names
        train_set = [columns[index] for index in train_columns]
        test_set = [columns[index] for index in test_columns]

        # Add the tuple (training, testing) to the result list
        result.append((train_set, test_set))

    # Save the resulting array to the specified file
    with open(file_name, 'wb') as f:
        pickle.dump(result, f)

    return result


def arr_5fold(columns=None, file_name='5folds.npy', available=True):

    if available:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data

    '''
    Parameters
    ----------
    columns : a list of dataframe's columns
        The total number of columns.
    '5folds.npy' : string
        where to store the resulting array.

    Returns
    -------
    a list of tuples. The index of the list refers to the validation fold and the elements of the tuples are the columns for training and testing respectively
    '''

    assert columns is not None

    # Calculate the number of columns in each fold
    num_columns = len(columns)
    # Create an array of indices representing the columns
    indices = np.arange(num_columns)

    # Randomly shuffle the indices
    np.random.shuffle(indices)

    # Split the indices into 5 equal parts (folds)
    folds = np.array_split(indices, 5)

    # Create a list to store the resulting tuples (training, testing)
    result = []

    # Iterate through each fold and create the training and testing sets
    for i in range(5):
        test_columns = folds[i]
        train_columns = np.concatenate([folds[j] for j in range(5) if j != i])

        # Convert the indices back to column names
        train_set = [columns[index] for index in train_columns]
        test_set = [columns[index] for index in test_columns]

        # Add the tuple (training, testing) to the result list
        result.append((train_set, test_set))

    # Save the resulting array to the specified file
    with open(file_name, 'wb') as f:
        pickle.dump(result, f)

    return result

def fold(id):
    '''
    

    Parameters
    ----------
    id : int
        the fold id from [1 to 10].

    Returns
    -------
    TYPE
        A tuple of the training columns and training columns for the fold.

    '''
    
    return arr_10fold()[id - 1]

def fold5(id):
    return arr_5fold()[id - 1]
    
'''
def add2dExamples(prev_array_location, new_examples, initial = False, return_list = False):
    
    if initial:
        loaded_array_list = []
        
        if isinstance(new_examples, list):
            # If it's a list, extend the loaded array list with the new examples
            loaded_array_list.extend(new_examples)
        else:
            # If it's a single 2D array, append it to the loaded array list
            loaded_array_list.append(new_examples)
        
        
        
        with open(prev_array_location, 'wb') as f:
            pickle.dump(loaded_array_list, f)
            
        return
    
    if return_list:
        with open(prev_array_location, 'rb') as f:
            loaded_array_list = pickle.load(f)
        
        return loaded_array_list
    
    with open(prev_array_location, 'rb') as f:
        loaded_array_list = pickle.load(f)
        
    
    # Check if new_examples is a list or a single 2D array
    if isinstance(new_examples, list):
        # If it's a list, extend the loaded array list with the new examples
        loaded_array_list.extend(new_examples)
    else:
        # If it's a single 2D array, append it to the loaded array list
        loaded_array_list.append(new_examples)
    
    with open(prev_array_location, 'wb') as f:
        pickle.dump(loaded_array_list, f)
'''


def add2dExamples(prev_array_location, new_examples=None, initial=False):
    """
    Adds new 2D numpy arrays to a list of 2D numpy arrays stored in a file, and returns the updated list.
    If the initial flag is set, it initializes a new empty list of 2D numpy arrays.
    
    Parameters
    ----------
    prev_array_location : str
        The file path where the list of 2D numpy arrays is stored, or will be stored if the initial flag is set.
    new_examples : list of 2D numpy arrays or single 2D numpy array, optional
        The new 2D numpy arrays to add to the existing list. Can be a list of 2D numpy arrays or a single 2D numpy array.
    initial : bool, optional, default=False
        If set to True, initializes a new empty list of 2D numpy arrays in the given file path.
    
    Returns
    -------
    list
        The updated list of 2D numpy arrays after adding new_examples, or the current list if no new_examples were provided.
    """
    # If the initial flag is set, create a new empty list
    if initial:
        loaded_array_list = []
    else:
        # Load the list of 2D numpy arrays from the file using pickle
        with open(prev_array_location, 'rb') as f:
            loaded_array_list = pickle.load(f)

    # If new_examples is provided, add them to the list
    if new_examples is not None:
        # Check if new_examples is a list or a single 2D array
        if isinstance(new_examples, list):
            # If it's a list, extend the loaded array list with the new examples
            loaded_array_list.extend(new_examples)
        else:
            # If it's a single 2D array, append it to the loaded array list
            loaded_array_list.append(new_examples)

        # Save the updated list of 2D numpy arrays to the file using pickle
        with open(prev_array_location, 'wb') as f:
            pickle.dump(loaded_array_list, f)

    return loaded_array_list



def select(df, consumer):
    '''
    This function takes a DataFrame and a consumer as input, and returns a new DataFrame containing only the specified consumer's data and the corresponding 'Year-Month' column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the consumer data and a 'Year-Month' column.
    consumer : str
        The name of the consumer whose data should be selected.

    Returns
    -------
    selected_df : pandas.DataFrame
        A new DataFrame containing the specified consumer's data and the corresponding 'Year-Month' column.
    '''
    return df[[consumer, 'Year-Month']]


import pandas as pd




def examples_from_file(indices ,file_name = 'attack name type or original'):
    index_values = [ *range(3615, 42372)]

    index = pd.Index(index_values, dtype='object')
    # load file
    with open(file_name, 'rb') as f:
        file = pickle.load(f)
    
    
    list_of_2darr = [w for i in indices if (w := file[index.get_loc(i)]) is not None]

    
    return np.vstack(list_of_2darr)



# def mainset(foldId):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    