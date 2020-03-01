import numpy as np
import torch


def to_torch_arg(array):
    if np.iscomplex(array).any():
        torch_x = np.stack((np.real(array), np.imag(array)))
        return torch.tensor(torch_x)
    else:
        return torch.tensor(array)
