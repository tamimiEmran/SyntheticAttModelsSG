# src/utils/seeding.py
"""
Utility function for setting random seeds for reproducibility.
"""

import random
import numpy as np
import os

# Optional: Add torch seed setting if PyTorch is used later
# try:
#     import torch
#     TORCH_AVAILABLE = True
# except ImportError:
#     TORCH_AVAILABLE = False

def set_seed(seed: int):
    """
    Sets the random seeds for Python, NumPy, and potentially other libraries
    to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # For consistent hashing

    # if TORCH_AVAILABLE:
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed)
    #         torch.cuda.manual_seed_all(seed) # if using multi-GPU
    #         # Potentially set deterministic algorithms (might impact performance)
    #         # torch.backends.cudnn.deterministic = True
    #         # torch.backends.cudnn.benchmark = False

    print(f"Set random seed to {seed}")
