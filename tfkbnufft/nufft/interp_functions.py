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
    distind = tf.cast(tf.round((tm - gridind) * L[:, None]), int_type)
    gridind = tf.cast(gridind, int_type)

    arr_ind = tf.zeros((M,), dtype=int_type)
    coef = tf.ones(M, dtype=table.dtype)

    for d in range(ndims):  # spatial dimension
        sliced_table = tf.gather_nd(table[d], (distind[d, :] + centers[d])[:, None])
        if conjcoef:
            coef = coef * tf.math.conj(sliced_table)
        else:
            coef = coef * sliced_table
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
        tf.cast(tf.floor(tm - numpoints[:, None] / 2.0), int_type)

    # initialize output array
    kdat = tf.zeros(
        shape=(griddat.shape[0], tm.shape[-1]),
        dtype=dtype,
    )

    # loop over offsets and take advantage of broadcasting
    for J in Jlist:
        coef, arr_ind = calc_coef_and_indices(
            tm, kofflist, J, table, centers, L, dims)

        # I don't need to expand on coil dimension since I use tf gather and not
        # gather_nd
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
    int_type = tf.int64

    # center of tables
    centers = tf.cast(tf.floor(numpoints * L / 2), int_type)

    # offset from k-space to first coef loc
    kofflist = 1 + \
        tf.cast(tf.floor(tm - numpoints[:, None] / 2.0), int_type)

    # initialize output array
    griddat = tf.zeros(
        shape=(kdat.shape[0], 2, tf.reduce_prod(dims)),
        dtype=dtype,
    )

    # loop over offsets and take advantage of numpy broadcasting
    for J in Jlist:
        coef, arr_ind = calc_coef_and_indices(
            tm, kofflist, J, table, centers, L, dims, conjcoef=True)

        updates = coef[None, ...] * kdat
        # TODO: change because the array of indexes was only in one dimension
        arr_ind = arr_ind
        tf.tensor_scatter_nd_add(griddat, arr_ind, updates)

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

    # extract interpolation params
    n_shift = interpob['n_shift']




    if interp_mats is None:
        dtype = interpob['table'][0].dtype
        grid_size = interpob['grid_size']
        numpoints = interpob['numpoints']
        ndims = om.shape[1]

        # convert to normalized freq locs
        # the frequencies are originally in [-pi; pi]
        # we put them in [-grid_size/2; grid_size/2]
        tm = tf.zeros(shape=om.shape, dtype=dtype)
        Jgen = []
        for i in range(ndims):
            gam = (2 * np.pi / grid_size[i])
            tm[:, i, :] = om[:, i, :] / gam
            # number of points to use for interpolation is numpoints
            Jgen.append(tf.range(numpoints[i]))
        # build an iterator for going over all J values
        # this might need some revamp in case we can't use itertools:
        # - either use a tf py function if possible
        # - or use the answers provided https://stackoverflow.com/questions/47132665/cartesian-product-in-tensorflow
        Jgen = list(itertools.product(*Jgen))
        Jgen = tf.convert_to_tensor(Jgen)
        # set up params if not using sparse mats
        params = {
            'dims': None,
            'table': interpob['table'],
            'numpoints': numpoints,
            'Jlist': Jgen,
            'table_oversamp': interpob['table_oversamp'],
        }

    y = []
    # run the table interpolator for each batch element
    # TODO: look into how to use tf.scan
    for b in range(x.shape[0]):
        if interp_mats is None:
            params['dims'] = tf.shape(x[b])[1:]
            # tm are the localized frequency locations
            # view(x.shape[1], 2, -1) allows to have the values of each point
            # on the grid in a list, (x.shape[1] is the number of coils and 2
            # is the imag dim)
            y.append(run_interp(tf.reshape(x[b], (x.shape[1], -1)), tm[b], params))
        else:
            # TODO: take care of this
            y.append(
                run_mat_interp(
                    x[b].view((x.shape[1], 2, -1)),
                    # TODO: change to complex interp_mats
                    interp_mats['real_interp_mats'][b],
                    interp_mats['imag_interp_mats'][b],
                )
            )

        # phase for fftshift
        y[-1] = y[-1] * tf.exp(1j * tf.linalg.matvec(om[b], n_shift))[None, ...]

    y = tf.stack(y)

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
