from sklearn.model_selection import train_test_split
from Loader import Loader
import numpy as np
    
        # testset = create_testset(foldId)
def shuffle_tuple(tup, percentage=1.0):
    x, y = tup
    assert len(x) == len(y), "The number of examples and labels must be the same"
    
    # Generate a permutation of indices
    permuted_indices = np.random.permutation(len(x))

    # Shuffle examples and labels using the generated permutation
    shuffled_x = x[permuted_indices]
    shuffled_y = y[permuted_indices]

    # Calculate the number of examples to keep based on the given percentage
    num_examples_to_keep = int(len(x) * percentage)

    # Slice the arrays to keep only the desired percentage of data
    sliced_x = shuffled_x[:num_examples_to_keep]
    sliced_y = shuffled_y[:num_examples_to_keep]

    return sliced_x, sliced_y.reshape(-1,1)
     
# constants for this experiments are:
attack_types = [*range(0, 13), 'ieee']
dataClass = Loader()
     
def load_attack_data(fold_id, dataset_type, attack_types = attack_types):
    """
    Load attack dataset and return the training, validation, and testing data based on the fold id and dataset type (real or synthetic).

    Parameters:
    fold_id (int): Fold ID in the range of 1 to 10, representing one of the 10 folds.
    dataset_type (str): Dataset type, either 'real' or 'synthetic'.

    Returns:
    tuple: A tuple containing three tuples: (train_X, train_y), (validation_X, validation_y), (test_X, test_y).
    """
    
    print('preparing monthly examples')
    dataClass.realVsSynth_Monthlyexamples(fold_id, attacktypes= attack_types)
    
    # Your implementation here
    if int(fold_id) == 0:
        raise Exception('fold id starts from 1')
    
    if dataset_type == 'synthetic':
        train = dataClass.synth_train_examples , dataClass.synth_train_labels
        
        train_x, train_y = train
        validation_size = 0.005  # You can set this to the desired proportion of the validation set
        
        train_x, validation_x, train_y, validation_y = train_test_split(
            train_x, train_y, test_size=validation_size, random_state=42
        )
        
        train = (train_x, train_y)
        validation = (validation_x, validation_y)
        
        test = dataClass.synth_test_examples , dataClass.synth_test_labels
        
    elif dataset_type == 'real':
        
        train = dataClass.real_train_examples, dataClass.real_train_labels
        
        train_x, train_y = train
        validation_size = 0.01  # You can set this to the desired proportion of the validation set
        
        train_x, validation_x, train_y, validation_y = train_test_split(
            train_x, train_y, test_size=validation_size, random_state=42
        )
        
        train = (train_x, train_y)
        validation = (validation_x, validation_y)
        
        test = dataClass.real_test_examples, dataClass.real_test_labels
        

        
        
    else:
        raise Exception("dataset_type must either be a str: 'synthetic' or 'real'")
    
    
    
    
    return train, validation, test
    
    

        
    
        
        
