import numpy as np
import torch


def to_torch_arg(array):
    if isinstance(array, list):
        return [to_torch_arg(a) for a in array]
    else:
        if np.iscomplex(array).any():
            torch_x = np.stack((np.real(array), np.imag(array)))
            return torch.tensor(torch_x)
        else:
            return torch.tensor(array)

def torch_to_numpy(array, complex_dim=None):
    if complex_dim is not None:
        assert array.shape[complex_dim] == 2
        return array.select(complex_dim, 0).numpy() + 1j * array.select(complex_dim, 1).numpy()
    else:
        return array.numpy()
