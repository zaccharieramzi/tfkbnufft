import numpy as np
from skimage.data import shepp_logan_phantom
import tensorflow as tf
import torch

from tfkbnufft.nufft.fft_functions import fft_functions as tf_fft_functions
from torchkbnufft.nufft import fft_functions as torch_fft_functions

def test_scale_and_fft_on_image_volume():
    # problem definition
    x = shepp_logan_phantom().astype(np.complex)
    im_size = x.shape
    scaling_coeffs = np.random.randn(*im_size, dtype='complex64')
    grid_size = [2*im_dim for im_dim in im_size]
    # torch computations
    torch_x = np.stack((np.real(x), np.imag(x)))
    torch_x = torch.tensor(torch_x).unsqueeze(0).unsqueeze(0)
    torch_scaling_coeffs = torch.tensor(
        np.stack((np.real(scaling_coeffs), np.imag(scaling_coeffs)))
    )
    res_torch = torch_fft_functions.scale_and_fft_on_image_volume(
        torch_x,
        torch_scaling_coeffs,
        grid_size,
        im_size,
        True,
    ).numpy()
    res_torch = res_torch[:, :, 0] + 1j *res_torch[:, :, 1]
    # tf computations
    res_tf = torch_fft_functions.scale_and_fft_on_image_volume(
        tf.convert_to_tensor(x)[None, None, ...],
        tf.convert_to_tensor(scaling_coeffs),
        grid_size,
        im_size,
        True,
    ).numpy()
    np.testing.assert_allclose(res_torch, res_tf)
