import numpy as np
import tensorflow as tf
import torch


def to_torch_arg(arg):
    if isinstance(arg, list):
        return [to_torch_arg(a) for a in arg]
    elif isinstance(arg, dict):
        return {k: to_torch_arg(v) for k, v  in arg.items()}
    else:
        if np.iscomplex(arg).any():
            torch_x = np.stack((np.real(arg), np.imag(arg)))
            return torch.tensor(torch_x)
        else:
            return torch.tensor(arg)

def to_tf_arg(arg):
    if isinstance(arg, list):
        return [to_tf_arg(a) for a in arg]
    elif isinstance(arg, dict):
        return {k: to_tf_arg(v) for k, v in arg.items()}
    else:
        return tf.convert_to_tensor(arg)

def torch_to_numpy(array, complex_dim=None):
    if complex_dim is not None:
        assert array.shape[complex_dim] == 2
        return array.select(complex_dim, 0).numpy() + 1j * array.select(complex_dim, 1).numpy()
    else:
        return array.numpy()
