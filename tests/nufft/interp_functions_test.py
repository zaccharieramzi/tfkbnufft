import numpy as np
import pytest
import tensorflow as tf
import torch
from torchkbnufft.nufft import interp_functions as torch_interp_functions
from torchkbnufft.nufft.utils import build_table

from tfkbnufft.nufft import interp_functions as tf_interp_functions
from ..utils import to_torch_arg, torch_to_numpy


@pytest.mark.parametrize('conjcoef', [True, False])
def test_calc_coef_and_indices(conjcoef):
    grid_size = np.array([800, 800])
    im_size = grid_size / 2
    n_samples = 324000
    # normalized frequency locations (i.e. between -grid_size/2 and grid_size/2)
    tm = np.random.uniform(-grid_size/2, grid_size/2, size=(n_samples, len(im_size))).T
    numpoints = np.array([6,] * len(im_size))
    kofflist = 1 + np.floor(tm - numpoints[:, None] / 2).astype(int)
    Jval = np.random.uniform(numpoints).astype(int)
    table_oversamp = 2**10
    L = np.array([table_oversamp for i in im_size])
    table = build_table(
        numpoints=numpoints,
        table_oversamp=L,
        grid_size=grid_size,
        im_size=im_size,
        ndims=len(im_size),
        order=(0,) * len(im_size),
        alpha=tuple(np.array(2.34) * np.array(numpoints)),
    )
    centers = np.floor(numpoints * L / 2).astype(int)
    args = [tm, kofflist, Jval, table, centers, L.astype(float), grid_size]
    torch_args = [to_torch_arg(arg) for arg in args] + [conjcoef]
    res_torch_coefs, res_torch_ind = torch_interp_functions.calc_coef_and_indices(*torch_args)
    res_torch_coefs = torch_to_numpy(res_torch_coefs, complex_dim=0)
    tf_args = [tf.convert_to_tensor(arg) for arg in args] + [conjcoef]
    res_tf_coefs, res_tf_ind = tf_interp_functions.calc_coef_and_indices(*tf_args)
    np.testing.assert_equal(res_torch_ind.numpy(), res_tf_ind.numpy())
    np.testing.assert_allclose(res_torch_coefs, res_tf_coefs.numpy())
