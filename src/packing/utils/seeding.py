import random
import numpy as np
import torch
import os
from transformers import set_seed

def generate_random_seed():
    """Generate a random seed."""
    return random.randint(0, 2**32 - 1)


# Update this function whenever you have a library that needs to be seeded.
def seed_everything(seed):
    """Seed all random generators."""
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # For transformers:
    set_seed(seed)
    #torch.use_deterministic_algorithms(True)

    # For numpy:
    # This is for legacy numpy:
    # np.random.seed(config.seed)
    # New code should make a Generator out of the config.seed directly:
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html

    # For PyTorch:
    # torch.manual_seed(config.seed)
    # Higher (e.g., on CUDA too) reproducibility with deterministic algorithms:
    # https://pytorch.org/docs/stable/notes/randomness.html
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Not supported for all operations though:
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html

