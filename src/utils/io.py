# src/utils/io.py
"""
Utility functions for input/output operations, like saving/loading objects.
"""

import pickle
import os
from typing import Any

def save_pickle(data: Any, filepath: str, protocol: int = pickle.HIGHEST_PROTOCOL):
    """
    Saves Python object data to a file using pickle.

    Args:
        data (Any): The Python object to save.
        filepath (str): The path to the file where the object will be saved.
        protocol (int): The pickle protocol to use.
    """
    try:
        # Ensure the directory exists
        dirpath = os.path.dirname(filepath)
        if dirpath: # Check if path includes a directory
            os.makedirs(dirpath, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)
        print(f"Successfully saved data to {filepath}")
    except IOError as e:
        print(f"Error: Could not write to file {filepath}. Reason: {e}")
        # Optionally re-raise the exception if saving is critical
        # raise
    except pickle.PicklingError as e:
        print(f"Error: Could not pickle the data for saving to {filepath}. Reason: {e}")
        # raise
    except Exception as e:
        print(f"An unexpected error occurred during saving to {filepath}: {e}")
        # raise


def load_pickle(filepath: str) -> Any:
    """
    Loads Python object data from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        Any: The loaded Python object.

    Raises:
        FileNotFoundError: If the filepath does not exist.
        Exception: If there is an error during loading or unpickling.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: Pickle file not found at {filepath}")

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded data from {filepath}")
        return data
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
        raise Exception(f"Error: Could not unpickle data from {filepath}. File might be corrupted or incompatible. Reason: {e}")
    except IOError as e:
        raise Exception(f"Error: Could not read file {filepath}. Reason: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during loading from {filepath}: {e}")
