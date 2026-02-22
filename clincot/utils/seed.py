from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: random seed.
        deterministic: if True, enable deterministic CUDA behavior when possible.
    """
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some ops may not support strict deterministic mode.
            pass
