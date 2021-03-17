import numpy as np
import pytest
from torchkbnufft.nufft import interp_functions as torch_interp_functions
from torchkbnufft.nufft.utils import build_table

from tfkbnufft.nufft import interp_functions as tf_interp_functions
from tfkbnufft.utils.itertools import cartesian_product
from ..utils import to_torch_arg, torch_to_numpy, to_tf_arg

def setup():
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
    tf_args = [to_tf_arg(arg) for arg in args] + [conjcoef]
    res_tf_coefs, res_tf_ind = tf_interp_functions.calc_coef_and_indices(*tf_args)
    np.testing.assert_equal(res_torch_ind.numpy(), res_tf_ind.numpy())
    np.testing.assert_allclose(res_torch_coefs, res_tf_coefs.numpy())

@pytest.mark.parametrize('n_coil', [1, 2, 5, 16])
@pytest.mark.parametrize('conjcoef', [True, False])
def test_run_interp(n_coil, conjcoef):
    tm, Jgen, table, numpoints, L, grid_size = setup()
    grid_size = grid_size.astype('int')
    griddat = np.stack([
        np.reshape(
            np.random.randn(*grid_size) + 1j * np.random.randn(*grid_size),
            [-1],
        ) for i in range(n_coil)
    ])
    params = {
        'dims': grid_size,
        'table': table,
        'numpoints': numpoints,
        'Jlist': Jgen,
        'table_oversamp': L,
        'conjcoef': conjcoef,
    }
    args = [griddat, tm, params]
    if not conjcoef:
        torch_args = [to_torch_arg(arg) for arg in args]
        # I need this because griddat is first n_coil then real/imag
        torch_args[0] = torch_args[0].permute(1, 0, 2)
        res_torch = torch_interp_functions.run_interp(*torch_args)
    # I need this because I create Jlist in a neater way for tensorflow
    params['Jlist'] = Jgen.T
    tf_args = [to_tf_arg(arg) for arg in args]
    res_tf = tf_interp_functions.run_interp(*tf_args)
    if not conjcoef:
        # Compare results with torch
        np.testing.assert_allclose(torch_to_numpy(res_torch, complex_dim=1), res_tf.numpy())

@pytest.mark.parametrize('n_coil', [1, 2, 5, 16])
def test_run_interp_back(n_coil):
    tm, Jgen, table, numpoints, L, grid_size = setup()
    num_samples = tm.shape[1]
    grid_size = grid_size.astype('int')
    kdat = np.stack([
        np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        for i in range(n_coil)
    ])
    params = {
        'dims': grid_size,
        'table': table,
        'numpoints': numpoints,
        'Jlist': Jgen,
        'table_oversamp': L,
    }
    args = [kdat, tm, params]
    torch_args = [to_torch_arg(arg) for arg in args]
    # I need this because griddat is first n_coil then real/imag
    torch_args[0] = torch_args[0].permute(1, 0, 2)
    res_torch = torch_interp_functions.run_interp_back(*torch_args)
    # I need this because I create Jlist in a neater way for tensorflow
    params['Jlist'] = Jgen.T
    tf_args = [to_tf_arg(arg) for arg in args]
    res_tf = tf_interp_functions.run_interp_back(*tf_args)
    np.testing.assert_allclose(torch_to_numpy(res_torch, complex_dim=1), res_tf.numpy())

@pytest.mark.parametrize('n_coil', [1, 2, 5, 16])
def test_kbinterp(n_coil):
    tm, _, table, numpoints, L, grid_size = setup()
    grid_size = grid_size.astype('int')
    x = np.stack([
        np.random.randn(*grid_size) + 1j * np.random.randn(*grid_size)
        for i in range(n_coil)
    ])[None, ...]  # adding batch dimension
    tm = tm[None, ...]  # adding batch dimension
    n_shift = np.array((grid_size//2) // 2).astype('float')
    Jgen = []
    for i in range(2):
        # number of points to use for interpolation is numpoints
        Jgen.append(np.arange(numpoints[i]))
    Jgen = cartesian_product(Jgen)
    interpob = {
        'grid_size': grid_size.astype('float'),
        'table': table,
        'numpoints': numpoints,
        'Jlist': Jgen.astype('int64'),
        'table_oversamp': L,
        'n_shift': n_shift,
    }
    om = np.zeros_like(tm)
    for i in range(tm.shape[1]):
        gam = (2 * np.pi / grid_size[i])
        om[:, i, :] = tm[:, i, :] * gam
    args = [x, om, interpob]
    torch_args = [to_torch_arg(arg) for arg in args]
    # I need this because griddat is first nbatch, n_coil then real/imag
    torch_args[0] = torch_args[0].permute(1, 2, 0, 3, 4)
    res_torch = torch_interp_functions.kbinterp(*torch_args)
    tf_args = [to_tf_arg(arg) for arg in args]
    res_tf = tf_interp_functions.kbinterp(*tf_args)
    # those tols seem like a lot, but for now it'll do
    np.testing.assert_allclose(torch_to_numpy(res_torch, complex_dim=2), res_tf.numpy(), rtol=1e-1, atol=2*1e-2)

@pytest.mark.parametrize('n_coil', [1, 2, 5, 16])
def test_adjkbinterp(n_coil):
    tm, _, table, numpoints, L, grid_size = setup()
    num_samples = tm.shape[1]
    grid_size = grid_size.astype('int')
    y = np.stack([
        np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        for i in range(n_coil)
    ])[None, ...]  # adding batch dimension
    tm = tm[None, ...]  # adding batch dimension
    n_shift = np.array((grid_size//2) // 2).astype('float')
    Jgen = []
    for i in range(2):
        # number of points to use for interpolation is numpoints
        Jgen.append(np.arange(numpoints[i]))
    Jgen = cartesian_product(Jgen)
    interpob = {
        'grid_size': grid_size.astype('float'),
        'table': table,
        'numpoints': numpoints,
        'Jlist': Jgen.astype('int64'),
        'table_oversamp': L,
        'n_shift': n_shift,
    }
    om = np.zeros_like(tm)
    for i in range(tm.shape[1]):
        gam = (2 * np.pi / grid_size[i])
        om[:, i, :] = tm[:, i, :] * gam
    args = [y, om, interpob]
    torch_args = [to_torch_arg(arg) for arg in args]
    # I need this because griddat is first nbatch, n_coil then real/imag
    torch_args[0] = torch_args[0].permute(1, 2, 0, 3)
    res_torch = torch_interp_functions.adjkbinterp(*torch_args)
    tf_args = [to_tf_arg(arg) for arg in args]
    res_tf = tf_interp_functions.adjkbinterp(*tf_args)
    # those tols seem like a lot, but for now it'll do
    np.testing.assert_allclose(torch_to_numpy(res_torch, complex_dim=2), res_tf.numpy(), rtol=1e-1, atol=2*1e-2)
