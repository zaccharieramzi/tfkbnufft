import numpy as np
import pytest
import tensorflow as tf
import torch
from torchkbnufft.nufft import interp_functions as torch_interp_functions
from torchkbnufft.nufft.utils import build_table

from tfkbnufft.nufft import interp_functions as tf_interp_functions
from ..utils import to_torch_arg, torch_to_numpy, to_tf_arg

def setup():
    # TODO: test with multicoils, i.e griddat = np.stack([griddat]*n_coils)
    grid_size = np.array([800, 800])
    im_size = grid_size / 2
    n_samples = 324000
    # normalized frequency locations (i.e. between -grid_size/2 and grid_size/2)
    tm = np.random.uniform(-grid_size/2, grid_size/2, size=(n_samples, len(im_size))).T
    numpoints = np.array([6,] * len(im_size))
    Jgen = np.array([np.random.uniform(numpoints).astype(int)]).T
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
    numpoints = numpoints.astype('float')
    L = L.astype('float')
    grid_size = grid_size.astype('float')
    return tm, Jgen, table, numpoints, L, grid_size


@pytest.mark.parametrize('conjcoef', [True, False])
def test_calc_coef_and_indices(conjcoef):
    tm, Jgen, table, numpoints, L, grid_size = setup()
    kofflist = 1 + np.floor(tm - numpoints[:, None] / 2).astype(int)
    Jval = Jgen[0]
    centers = np.floor(numpoints * L / 2).astype(int)
    args = [tm, kofflist, Jval, table, centers, L, grid_size.astype('int')]
    torch_args = [to_torch_arg(arg) for arg in args] + [conjcoef]
    res_torch_coefs, res_torch_ind = torch_interp_functions.calc_coef_and_indices(*torch_args)
    res_torch_coefs = torch_to_numpy(res_torch_coefs, complex_dim=0)
    tf_args = [tf.convert_to_tensor(arg) for arg in args] + [conjcoef]
    res_tf_coefs, res_tf_ind = tf_interp_functions.calc_coef_and_indices(*tf_args)
    np.testing.assert_equal(res_torch_ind.numpy(), res_tf_ind.numpy())
    np.testing.assert_allclose(res_torch_coefs, res_tf_coefs.numpy())

def test_run_interp():
    tm, Jgen, table, numpoints, L, grid_size = setup()
    griddat = np.reshape(
        np.random.randn(*grid_size.astype('int')) + 1j * np.random.randn(*grid_size.astype('int')),
        (1, -1),  # 1 coil
    )
    params = {
        'dims': grid_size.astype('int'),
        'table': table,
        'numpoints': numpoints,
        'Jlist': Jgen,
        'table_oversamp': L,
    }
    args = [griddat, tm, params]
    torch_args = [to_torch_arg(arg) for arg in args]
    torch_args[0] = torch_args[0].permute(1, 0, 2)
    res_torch = torch_interp_functions.run_interp(*torch_args)
    params['Jlist'] = Jgen.T
    tf_args = [to_tf_arg(arg) for arg in args]
    res_tf = tf_interp_functions.run_interp(*tf_args)
    np.testing.assert_allclose(torch_to_numpy(res_torch, complex_dim=1), res_tf.numpy())
