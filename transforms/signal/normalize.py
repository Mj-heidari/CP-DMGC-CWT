import torch
import numpy as np

class InstanceNormTransform:
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Normalize per channel across time (common in EEG)
        # shape expected: (channels, time)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / (std + self.eps)

        return x.numpy()  # keep dataset outputs consistent