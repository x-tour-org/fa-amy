import os
import random

import numpy as np
import torch


class Seeder:

    @staticmethod
    def set_seed(seed: int = 777):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"