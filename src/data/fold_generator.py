# src/data/fold_generator.py
"""
Functions for creating, saving, loading, and retrieving k-fold cross-validation splits.
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Union, Any, Optional

def create_folds(
    items: List[Any],
    n_folds: int,
    filename: str,
    random_state: Optional[int] = 42,
    shuffle: bool = True,
    overwrite: bool = False
) -> List[Tuple[List[Any], List[Any]]]:
    """
    Creates k-fold cross-validation splits for a list of items (e.g., consumer IDs),
    saves them to a file, and returns the folds.

    Args:
        items (List[Any]): The list of items to split into folds.
        n_folds (int): The number of folds (k).
        filename (str): Path to the file where the folds will be saved (e.g., '10folds.pkl').
        random_state (Optional[int]): Seed for the random number generator for shuffling.
        shuffle (bool): Whether to shuffle the items before splitting.
        overwrite (bool): If True, overwrite the file if it exists. Otherwise, load existing.

    Returns:
        List[Tuple[List[Any], List[Any]]]: A list of length n_folds. Each element is a tuple
                                          (train_items, test_items) for that fold.

    Raises:
        ValueError: If n_folds is less than 2 or greater than the number of items.
    """
    if not overwrite and os.path.exists(filename):
        print(f"Loading existing folds from {filename}")
        return load_folds(filename)

    if n_folds < 2 or n_folds > len(items):
        raise ValueError("n_folds must be between 2 and the number of items.")

    num_items = len(items)
    indices = np.arange(num_items)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    # Split indices into n_folds parts
    # np.array_split handles cases where num_items is not perfectly divisible by n_folds
    fold_indices = np.array_split(indices, n_folds)

    result_folds = []
    item_array = np.array(items) # Convert to numpy array for easier indexing

    for i in range(n_folds):
        test_idx = fold_indices[i]
        # Concatenate indices from all other folds for training set
        train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])

        # Get the actual items using the indices
        train_set_items = item_array[train_idx].tolist()
        test_set_items = item_array[test_idx].tolist()

        result_folds.append((train_set_items, test_set_items))

    # Save the resulting list of folds using pickle
    try:
        with open(filename, 'wb') as f:
            pickle.dump(result_folds, f)
        print(f"Saved {n_folds} folds to {filename}")
    except IOError as e:
        print(f"Error saving folds to {filename}: {e}")
        # Decide if we should still return result_folds or raise the error

    return result_folds

def load_folds(filename: str) -> List[Tuple[List[Any], List[Any]]]:
    """
    Loads k-fold splits from a pickle file.

    Args:
        filename (str): Path to the file containing the saved folds.

    Returns:
        List[Tuple[List[Any], List[Any]]]: The loaded list of folds.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For issues during unpickling.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Folds file not found: {filename}")

    try:
        with open(filename, 'rb') as f:
            folds = pickle.load(f)
        # Basic validation
        if not isinstance(folds, list) or not all(isinstance(f, tuple) and len(f) == 2 for f in folds):
             raise ValueError(f"File {filename} does not contain a valid list of fold tuples.")
        return folds
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
        raise Exception(f"Error loading or unpickling folds from {filename}: {e}")
    except IOError as e:
        raise Exception(f"Error reading file {filename}: {e}")


def get_fold(fold_id: int, filename: str) -> Tuple[List[Any], List[Any]]:
    """
    Retrieves the training and testing items for a specific fold ID (1-based index).

    Args:
        fold_id (int): The fold number (starting from 1).
        filename (str): Path to the file containing the saved folds.

    Returns:
        Tuple[List[Any], List[Any]]: A tuple (train_items, test_items) for the specified fold.

    Raises:
        ValueError: If fold_id is out of bounds.
        FileNotFoundError: If the folds file does not exist.
    """
    folds = load_folds(filename)
    n_folds = len(folds)

    if not 1 <= fold_id <= n_folds:
        raise ValueError(f"fold_id must be between 1 and {n_folds}, but got {fold_id}.")

    # Adjust fold_id to be 0-based index for list access
    return folds[fold_id - 1]


# Example Usage (commented out)
# if __name__ == "__main__":
#     consumer_ids = [f'cons_{i}' for i in range(100)]
#     folds_file = 'my_10_folds.pkl'
#     num_folds = 10

#     # Create folds (will save if file doesn't exist or overwrite=True)
#     try:
#         created_folds = create_folds(consumer_ids, num_folds, folds_file, overwrite=False)
#         print(f"\nCreated/Loaded {len(created_folds)} folds.")

#         # Get fold 3
#         train_items_f3, test_items_f3 = get_fold(3, folds_file)
#         print(f"\nFold 3:")
#         print(f"  Train items count: {len(train_items_f3)}")
#         print(f"  Test items count: {len(test_items_f3)}")
#         print(f"  Example Test item: {test_items_f3[0] if test_items_f3 else 'N/A'}")

#         # Verify no overlap
#         overlap = set(train_items_f3) & set(test_items_f3)
#         print(f"  Overlap between train/test: {len(overlap)}")

#         # Verify all items are covered
#         all_items_in_fold3 = set(train_items_f3) | set(test_items_f3)
#         print(f"  All items covered: {len(all_items_in_fold3) == len(consumer_ids)}")

#     except (FileNotFoundError, ValueError, Exception) as e:
#         print(f"\nAn error occurred: {e}")
