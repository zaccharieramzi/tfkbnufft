import numpy as np
import pytest
from skimage.data import shepp_logan_phantom
import tensorflow as tf
import torch

from tfkbnufft.nufft import fft_functions as tf_fft_functions
from torchkbnufft.nufft import fft_functions as torch_fft_functions

def _crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

@pytest.mark.parametrize('norm', ['ortho', None])
@pytest.mark.parametrize('multiprocessing', [True, False])
def test_scale_and_fft_on_image_volume(norm, multiprocessing):
    # problem definition
    x = shepp_logan_phantom().astype(np.complex64)
    im_size = x.shape
    scaling_coeffs = np.random.randn(*im_size) + 1j * np.random.randn(*im_size)
    scaling_coeffs = scaling_coeffs.astype(np.complex64)
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
        torch.tensor(grid_size).float(),
        torch.tensor(im_size),
        norm,

    ).numpy()
    res_torch = res_torch[:, :, 0] + 1j *res_torch[:, :, 1]
    # tf computations
    res_tf = tf_fft_functions.scale_and_fft_on_image_volume(
        tf.convert_to_tensor(x)[None, None, ...],
        tf.convert_to_tensor(scaling_coeffs),
        tf.convert_to_tensor(grid_size),
        tf.convert_to_tensor(im_size),
        norm,
        multiprocessing=multiprocessing,
    ).numpy()
    np.testing.assert_allclose(res_torch, res_tf, rtol=1e-4, atol=2*1e-2)

@pytest.mark.parametrize('norm', ['ortho', None])
@pytest.mark.parametrize('multiprocessing', [True, False])
def test_ifft_and_scale_on_gridded_data(norm, multiprocessing):
    # problem definition
    x = shepp_logan_phantom().astype(np.complex64)
    grid_size = x.shape
    im_size = [im_dim//2 for im_dim in grid_size]
    scaling_coeffs = np.random.randn(*im_size) + 1j * np.random.randn(*im_size)
    scaling_coeffs = scaling_coeffs.astype(np.complex64)
    # torch computations
    torch_x = np.stack((np.real(x), np.imag(x)))
    torch_x = torch.tensor(torch_x).unsqueeze(0).unsqueeze(0)
    torch_scaling_coeffs = torch.tensor(
        np.stack((np.real(scaling_coeffs), np.imag(scaling_coeffs)))
    )
    res_torch = torch_fft_functions.ifft_and_scale_on_gridded_data(
        torch_x,
        torch_scaling_coeffs,
        torch.tensor(grid_size).float(),
        torch.tensor(im_size),
        norm,
    ).numpy()
    res_torch = res_torch[:, :, 0] + 1j *res_torch[:, :, 1]
    # tf computations
    res_tf = tf_fft_functions.ifft_and_scale_on_gridded_data(
        tf.convert_to_tensor(x)[None, None, ...],
        tf.convert_to_tensor(scaling_coeffs),
        tf.convert_to_tensor(grid_size),
        tf.convert_to_tensor(im_size),
        norm,
        multiprocessing=multiprocessing,
    ).numpy()
    np.testing.assert_allclose(res_torch, res_tf, rtol=1e-4, atol=2)

@pytest.mark.parametrize('norm', ['ortho', None])
@pytest.mark.parametrize('multiprocessing', [True, False])
def test_scale_and_fft_on_image_volume_3d(norm, multiprocessing):
    # problem definition
    x = shepp_logan_phantom().astype(np.complex64)
    x = _crop_center(x, 128, 128)
    x = x[None, ...]
    x = np.tile(x, [128, 1, 1])
    im_size = x.shape
    scaling_coeffs = np.random.randn(*im_size) + 1j * np.random.randn(*im_size)
    scaling_coeffs = scaling_coeffs.astype(np.complex64)
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
        torch.tensor(grid_size).float(),
        torch.tensor(im_size),
        norm,

    ).numpy()
    res_torch = res_torch[:, :, 0] + 1j *res_torch[:, :, 1]
    # tf computations
    res_tf = tf_fft_functions.scale_and_fft_on_image_volume(
        tf.convert_to_tensor(x)[None, None, ...],
        tf.convert_to_tensor(scaling_coeffs),
        tf.convert_to_tensor(grid_size),
        tf.convert_to_tensor(im_size),
        norm,
        im_rank=3,
        multiprocessing=multiprocessing,
    ).numpy()
    np.testing.assert_allclose(res_torch, res_tf, rtol=1e-4, atol=2*1e-2)

@pytest.mark.parametrize('norm', ['ortho', None])
@pytest.mark.parametrize('multiprocessing', [True, False])
def test_ifft_and_scale_on_gridded_data_3d(norm, multiprocessing):
    # problem definition
    x = shepp_logan_phantom().astype(np.complex64)
    x = _crop_center(x, 128, 128)
    x = x[None, ...]
    x = np.tile(x, [128, 1, 1])
    grid_size = x.shape
    im_size = [im_dim//2 for im_dim in grid_size]
    scaling_coeffs = np.random.randn(*im_size) + 1j * np.random.randn(*im_size)
    scaling_coeffs = scaling_coeffs.astype(np.complex64)
    # torch computations
    torch_x = np.stack((np.real(x), np.imag(x)))
    torch_x = torch.tensor(torch_x).unsqueeze(0).unsqueeze(0)
    torch_scaling_coeffs = torch.tensor(
        np.stack((np.real(scaling_coeffs), np.imag(scaling_coeffs)))
    )
    res_torch = torch_fft_functions.ifft_and_scale_on_gridded_data(
        torch_x,
        torch_scaling_coeffs,
        torch.tensor(grid_size).float(),
        torch.tensor(im_size),
        norm,
    ).numpy()
    res_torch = res_torch[:, :, 0] + 1j *res_torch[:, :, 1]
    # tf computations
    res_tf = tf_fft_functions.ifft_and_scale_on_gridded_data(
        tf.convert_to_tensor(x)[None, None, ...],
        tf.convert_to_tensor(scaling_coeffs),
        tf.convert_to_tensor(grid_size),
        tf.convert_to_tensor(im_size),
        norm,
        im_rank=3,
        multiprocessing=multiprocessing,
    ).numpy()
    np.testing.assert_allclose(res_torch, res_tf, rtol=1e-4, atol=2)
