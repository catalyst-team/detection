import numpy as np
import torch


def detach(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
