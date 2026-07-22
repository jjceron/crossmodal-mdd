"""Seed management utilities for reproducible experiments."""

import os
import numpy as np
import random
import torch


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def parse_seeds(seed_args):
    if len(seed_args) == 1:
        return [seed_args[0]]
    return seed_args
