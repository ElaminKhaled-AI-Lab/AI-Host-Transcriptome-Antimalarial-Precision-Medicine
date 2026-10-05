"""Global seed control for reproducibility (blueprint Sections 70, 74, 90)."""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs.

    Also sets PYTHONHASHSEED and, for torch, deterministic cuDNN so repeated
    runs on the same seed reproduce splits and initialisation.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
