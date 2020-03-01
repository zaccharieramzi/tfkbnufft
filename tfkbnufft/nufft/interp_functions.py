import itertools

import numpy as np
import tensorflow as tf


def run_mat_interp(griddat, coef_mat):
    """Interpolates griddat to off-grid coordinates with input sparse matrices.

    Args:
        griddat (tensor): The gridded frequency data.
        coef_mat (sparse tensor): The interpolation coefficients
            stored as a sparse tensor.

    Returns:
        tensor: griddat interpolated to off-grid locations.
    """
    # TODO: look into sparse ordering for optimality of these mat mul
    kdat = tf.sparse.sparse_dense_matmul(coef_mat, tf.transpose(griddat))
    return kdat


def run_mat_interp_back(kdat, coef_mat):
    """Interpolates kdat to on-grid coordinates with input sparse matrices.

    Args:
        kdat (tensor): The off-grid frequency data.
        coef_mat (sparse tensor): The interpolation coefficients
            stored as a sparse tensor.

    Returns:
        tensor: kdat interpolated to on-grid locations.
    """
    griddat = tf.sparse.sparse_dense_matmul(tf.math.conj(coef_mat), tf.transpose(kdat))

    return griddat


def calc_coef_and_indices(tm, kofflist, Jval, table, centers, L, dims, conjcoef=False):
    """Calculates interpolation coefficients and on-grid indices.

    Args:
        tm (tensor): normalized frequency locations.
        kofflist (tensor): A tensor with offset locations to first elements in
            list of nearest neighbords.
        Jval (tensor): A tuple-like tensor for how much to increment offsets.
        table (list): A list of tensors tabulating a Kaiser-Bessel
            interpolation kernel.
        centers (tensor): A tensor with the center locations of the table for
            each dimension.
        L (tensor): A tensor with the table size in each dimension.
        dims (tensor): A tensor with image dimensions.
        conjcoef (boolean, default=False): A boolean for whether to compute
            normal or complex conjugate interpolation coefficients
            (conjugate needed for adjoint).

    Returns:
        tuple: A tuple with interpolation coefficients and indices.
    """
    # type values
    dtype = tm.dtype
    int_type = tf.int64

    # array shapes
    M = tm.shape[1]
    ndims = tm.shape[0]

    # indexing locations
    gridind = tf.cast(kofflist + Jval[:, None], dtype)
    distind = tf.cast(
        tf.round((tm - gridind) * L[:, None]),
        int_type,
    )
    gridind = tf.cast(gridind, int_type)

    arr_ind = tf.zeros((M,), dtype=int_type)
    coef = tf.stack((
        tf.ones(M, dtype=dtype),
        tf.zeros(M, dtype=dtype)
    ))

    for d in range(ndims):  # spatial dimension
        if conjcoef:
            coef = coef * table[d][:, distind[d, :] + centers[d]]
        else:
            coef = coef * tf.math.conj(table[d][:, distind[d, :] + centers[d]])
        arr_ind = arr_ind + tf.reshape(tf.math.floormod(gridind[d, :], dims[d]), [-1]) * \
            tf.reduce_prod(dims[d + 1:])

    return coef, arr_ind


def run_interp(griddat, tm, params):
    """Interpolates griddat to off-grid coordinates with input sparse matrices.

    Args:
        griddat (tensor): The on-grid frequency data.
        tm (tensor): Normalized frequency coordinates.
        params (dict): Dictionary with elements 'dims', 'table', 'numpoints',
            'Jlist', and 'table_oversamp'.

    Returns:
        tensor: griddat interpolated to off-grid locations.
    """
    # extract parameters
    dims = params['dims']
    table = params['table']
    numpoints = params['numpoints']
    Jlist = params['Jlist']
    L = params['table_oversamp']

    # extract data types
    dtype = table[0].dtype
    int_type = tf.int64

    # center of tables
    centers = tf.cast(tf.floor(numpoints * L / 2), int_type)

    # offset from k-space to first coef loc
    kofflist = 1 + \
        tf.cast(tf.floor(tm - numpoints.unsqueeze(1) / 2.0), int_type)

    # initialize output array
    kdat = tf.zeros(
        shape=(griddat.shape[0], tm.shape[-1]),
        dtype=dtype,
    )

    # loop over offsets and take advantage of broadcasting
    for Jind in range(Jlist.shape[1]):
        coef, arr_ind = calc_coef_and_indices(
            tm, kofflist, Jlist[:, Jind], table, centers, L, dims)

        # unsqueeze coil and real/imag dimensions for on-grid indices
        arr_ind = arr_ind[None, ...].tile([
            kdat.shape[0],
            1
        ])

        # gather and multiply coefficients
        kdat += coef[None, ...] * tf.gather(griddat, arr_ind, axis=1)

    return kdat


def run_interp_back(kdat, tm, params):
    """Interpolates kdat to on-grid coordinates.

    Args:
        kdat (tensor): The off-grid frequency data.
        tm (tensor): Normalized frequency coordinates.
        params (dict): Dictionary with elements 'dims', 'table', 'numpoints',
            'Jlist', and 'table_oversamp'.

    Returns:
        tensor: kdat interpolated to on-grid locations.
    """
    # extract parameters
    dims = params['dims']
    table = params['table']
    numpoints = params['numpoints']
    Jlist = params['Jlist']
    L = params['table_oversamp']

    # extract data types
    dtype = table[0].dtype
    device = table[0].device
    int_type = torch.long

    # center of tables
    centers = torch.floor(numpoints * L / 2).to(dtype=int_type)

    # offset from k-space to first coef loc
    kofflist = 1 + \
        torch.floor(tm - numpoints.unsqueeze(1) / 2.0).to(dtype=torch.long)

    # initialize output array
    griddat = torch.zeros(size=(kdat.shape[0], 2, torch.prod(dims)),
                          dtype=dtype, device=device)

    # loop over offsets and take advantage of numpy broadcasting
    for Jind in range(Jlist.shape[1]):
        coef, arr_ind = calc_coef_and_indices(
            tm, kofflist, Jlist[:, Jind], table, centers, L, dims, conjcoef=True)

        # the following code takes ordered data and scatters it on to an image grid
        # profiling for a 2D problem showed drastic differences in performances
        # for these two implementations on cpu/gpu, but they do the same thing
        if device == torch.device('cpu'):
            tmp = complex_mult(coef.unsqueeze(0), kdat, dim=1)
            for bind in range(griddat.shape[0]):
                for riind in range(griddat.shape[1]):
                    griddat[bind, riind].index_put_(
                        tuple(arr_ind.unsqueeze(0)),
                        tmp[bind, riind],
                        accumulate=True
                    )
        else:
            griddat.index_add_(
                2,
                arr_ind,
                complex_mult(coef.unsqueeze(0), kdat, dim=1)
            )

    return griddat


def kbinterp(x, om, interpob, interp_mats=None):
    """Apply table interpolation.

    Inputs are assumed to be batch/chans x coil x real/imag x image dims.
    Om should be nbatch x ndims x klength.

    Args:
        x (tensor): The oversampled DFT of the signal.
        om (tensor, optional): A custom set of k-space points to
            interpolate to in radians/voxel.
        interpob (dict): An interpolation object with 'table', 'n_shift',
            'grid_size', 'numpoints', and 'table_oversamp' keys. See
            models.kbinterp.py for details.
        interp_mats (dict, default=None): A dictionary with keys
            'real_interp_mats' and 'imag_interp_mats', each key containing a
            list of interpolation matrices (see
            mri.sparse_interp_mat.precomp_sparse_mats for construction). If
            None, then a standard interpolation is run.

    Returns:
        tensor: The signal interpolated to off-grid locations.
    """
    dtype = interpob['table'][0].dtype
    device = interpob['table'][0].device

    # extract interpolation params
    n_shift = interpob['n_shift']
    grid_size = interpob['grid_size']
    numpoints = interpob['numpoints']

    ndims = om.shape[1]

    # convert to normalized freq locs
    tm = torch.zeros(size=om.shape, dtype=dtype, device=device)
    Jgen = []
    for i in range(ndims):
        gam = (2 * np.pi / grid_size[i])
        tm[:, i, :] = om[:, i, :] / gam
        Jgen.append(range(np.array(numpoints[i].cpu(), dtype=np.int)))

    # build an iterator for going over all J values
    Jgen = list(itertools.product(*Jgen))
    Jgen = torch.tensor(Jgen).permute(1, 0).to(dtype=torch.long, device=device)

    if interp_mats is None:
        # set up params if not using sparse mats
        params = {
            'dims': None,
            'table': interpob['table'],
            'numpoints': numpoints,
            'Jlist': Jgen,
            'table_oversamp': interpob['table_oversamp'],
        }
    else:
        # make sure we're on the right device
        for real_mat in interp_mats['real_interp_mats']:
            assert real_mat.device == device
        for imag_mat in interp_mats['imag_interp_mats']:
            assert imag_mat.device == device

    y = []
    # run the table interpolator for each batch element
    for b in range(x.shape[0]):
        if interp_mats is None:
            params['dims'] = torch.tensor(
                x[b].shape[2:], dtype=torch.long, device=device)

            y.append(run_interp(x[b].view((x.shape[1], 2, -1)), tm[b], params))
        else:
            y.append(
                run_mat_interp(
                    x[b].view((x.shape[1], 2, -1)),
                    interp_mats['real_interp_mats'][b],
                    interp_mats['imag_interp_mats'][b],
                )
            )

        # phase for fftshift
        y[-1] = complex_mult(
            y[-1],
            imag_exp(torch.mv(torch.transpose(
                om[b], 1, 0), n_shift)).unsqueeze(0),
            dim=1
        )

    y = torch.stack(y)

    return y


def adjkbinterp(y, om, interpob, interp_mats=None):
    """Apply table interpolation adjoint.

    Inputs are assumed to be batch/chans x coil x real/imag x kspace length.
    Om should be nbatch x ndims x klength.

    Args:
        y (tensor): The off-grid DFT of the signal.
        om (tensor, optional): A set of k-space points to
            interpolate from in radians/voxel.
        interpob (dict): An interpolation object with 'table', 'n_shift',
            'grid_size', 'numpoints', and 'table_oversamp' keys. See
            models.kbinterp.py for details.
        interp_mats (dict, default=None): A dictionary with keys
            'real_interp_mats' and 'imag_interp_mats', each key containing a
            list of interpolation matrices (see
            mri.sparse_interp_mat.precomp_sparse_mats for construction). If
            None, then a standard interpolation is run.

    Returns:
        tensor: The signal interpolated to on-grid locations.
    """
    y = y.clone()

    n_shift = interpob['n_shift']
    grid_size = interpob['grid_size']
    numpoints = interpob['numpoints']

    dtype = interpob['table'][0].dtype
    device = interpob['table'][0].device

    ndims = om.shape[1]

    # convert to normalized freq locs
    tm = torch.zeros(size=om.shape, dtype=dtype, device=device)
    Jgen = []
    for i in range(ndims):
        gam = 2 * np.pi / grid_size[i]
        tm[:, i, :] = om[:, i, :] / gam
        Jgen.append(range(np.array(numpoints[i].cpu(), np.int)))

    # build an iterator for going over all J values
    Jgen = list(itertools.product(*Jgen))
    Jgen = torch.tensor(Jgen).permute(1, 0).to(dtype=torch.long, device=device)

    if interp_mats is None:
        # set up params if not using sparse mats
        params = {
            'dims': None,
            'table': interpob['table'],
            'numpoints': numpoints,
            'Jlist': Jgen,
            'table_oversamp': interpob['table_oversamp'],
        }
    else:
        # make sure we're on the right device
        for real_mat in interp_mats['real_interp_mats']:
            assert real_mat.device == device
        for imag_mat in interp_mats['imag_interp_mats']:
            assert imag_mat.device == device

    x = []
    # run the table interpolator for each batch element
    for b in range(y.shape[0]):
        # phase for fftshift
        y[b] = conj_complex_mult(
            y[b],
            imag_exp(torch.mv(torch.transpose(
                om[b], 1, 0), n_shift)).unsqueeze(0),
            dim=1
        )

        if interp_mats is None:
            params['dims'] = grid_size.to(dtype=torch.long, device=device)

            x.append(run_interp_back(y[b], tm[b], params))
        else:
            x.append(
                run_mat_interp_back(
                    y[b],
                    interp_mats['real_interp_mats'][b],
                    interp_mats['imag_interp_mats'][b],
                )
            )

    x = torch.stack(x)

    bsize = y.shape[0]
    ncoil = y.shape[1]
    out_size = (bsize, ncoil, 2) + tuple(grid_size.to(torch.long))

    x = x.view(out_size)

    return x