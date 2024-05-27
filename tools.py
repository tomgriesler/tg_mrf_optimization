"""
Tom Griesler, 05/2024
tomgr@umich.edu
"""

import torch

def to_tensor(x, dtype):
    return x if torch.is_tensor(x) else torch.tensor(x, dtype=dtype)