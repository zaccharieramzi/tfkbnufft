import math as m

import tensorflow as tf

from ..utils.itertools import product


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
    coef = tf.ones(M, dtype=table[0].dtype)

    for d in range(ndims):  # spatial dimension
        sliced_table = tf.gather_nd(table[d], (distind[d, :] + centers[d])[:, None])
        if conjcoef:
            coef = coef * tf.math.conj(sliced_table)
        else:
            coef = coef * sliced_table
        arr_ind = arr_ind + tf.reshape(tf.math.floormod(gridind[d, :], dims[d]), [-1]) * \
            tf.reduce_prod(dims[d + 1:])

    return coef, arr_ind

@tf.function
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
    L = tf.cast(L, tm.dtype)

    # extract data types
    int_type = tf.int64

    # center of tables
    centers = tf.cast(tf.floor(numpoints * L / 2), int_type)

    # offset from k-space to first coef loc
    kofflist = 1 + \
        tf.cast(tf.floor(tm - numpoints[:, None] / 2.0), int_type)

    # initialize output array
    kdat = tf.zeros(
        shape=(tf.shape(griddat)[0], tf.shape(tm)[-1]),
        dtype=griddat.dtype,
    )

    # loop over offsets and take advantage of broadcasting
    for J in Jlist:
        coef, arr_ind = calc_coef_and_indices(
            tm, kofflist, J, table, centers, L, dims)
        coef = tf.cast(coef, griddat.dtype)
        # I don't need to expand on coil dimension since I use tf gather and not
        # gather_nd
        # gather and multiply coefficients
        kdat += coef[None, ...] * tf.gather(griddat, arr_ind, axis=1)

    return kdat

@tf.function
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
    L = tf.cast(L, tm.dtype)

    # extract data types
    int_type = tf.int64

    # center of tables
    centers = tf.cast(tf.floor(numpoints * L / 2), int_type)

    # offset from k-space to first coef loc
    kofflist = 1 + \
        tf.cast(tf.floor(tm - numpoints[:, None] / 2.0), int_type)

    # initialize output array
    griddat = tf.zeros(
        shape=(tf.shape(kdat)[0], tf.cast(tf.reduce_prod(dims), tf.int32)),
        dtype=kdat.dtype,
    )

    # loop over offsets and take advantage of numpy broadcasting
    for J in Jlist:
        coef, arr_ind = calc_coef_and_indices(
            tm, kofflist, J, table, centers, L, dims, conjcoef=True)
        coef = tf.cast(coef, kdat.dtype)
        updates = tf.transpose(coef[None, ...] * kdat)
        # TODO: change because the array of indexes was only in one dimension
        arr_ind = arr_ind[:, None]
        griddat = tf.transpose(tf.tensor_scatter_nd_add(tf.transpose(griddat), arr_ind, updates))

    return griddat

@tf.function
def kbinterp(x, om, interpob):
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

    Returns:
        tensor: The signal interpolated to off-grid locations.
    """
    # extract interpolation params
    n_shift = interpob['n_shift']
    n_shift = tf.cast(n_shift, om.dtype)
    im_rank = interpob.get('im_rank', 2)
    # TODO: refactor all of this with adjkbinterp
    grid_size = interpob['grid_size']
    grid_size = tf.cast(grid_size, om.dtype)
    numpoints = interpob['numpoints']
    numpoints = tf.cast(numpoints, om.dtype)
    ndims = om.shape[1]

    # convert to normalized freq locs
    # the frequencies are originally in [-pi; pi]
    # we put them in [-grid_size/2; grid_size/2]
    pi = tf.constant(m.pi)
    tm = om * grid_size[None, :, None] / tf.cast(2 * pi, om.dtype)
    Jgen = []
    for i in range(ndims):
        # number of points to use for interpolation is numpoints
        Jgen.append(tf.range(numpoints[i]))
    # build an iterator for going over all J values
    Jgen = product(Jgen, im_rank)
    Jgen = tf.cast(Jgen, 'int64')
    # set up params if not using sparse mats
    params = {
        'dims': None,
        'table': interpob['table'],
        'numpoints': numpoints,
        'Jlist': Jgen,
        'table_oversamp': interpob['table_oversamp'],
    }

    y = tf.TensorArray(x.dtype, tf.shape(x)[0])
    y_not_shifted = tf.zeros(
        shape=(tf.shape(x)[1], tf.shape(tm)[-1]),
        dtype=x.dtype,
    )
    # run the table interpolator for each batch element
    # TODO: look into how to use tf.while_loop
    for b in tf.range(tf.shape(x)[0]):
        params['dims'] = tf.cast(tf.shape(x[b])[1:], 'int64')
        # tm are the localized frequency locations
        # view(x.shape[1], 2, -1) allows to have the values of each point
        # on the grid in a list, (x.shape[1] is the number of coils and 2
        # is the imag dim)
        y_not_shifted = run_interp(tf.reshape(x[b], (tf.shape(x)[1], -1)), tm[b], params)

        # phase for fftshift
        y = y.write(
            b,
            y_not_shifted * tf.exp(1j * tf.cast(tf.linalg.matvec(tf.transpose(om[b]), n_shift), y_not_shifted.dtype))[None, ...],
        )

    y = y.stack()

    return y

@tf.function
def adjkbinterp(y, om, interpob):
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

    Returns:
        tensor: The signal interpolated to on-grid locations.
    """
    n_shift = interpob['n_shift']
    n_shift = tf.cast(n_shift, om.dtype)
    im_rank = interpob.get('im_rank', 2)

    # TODO: refactor with kbinterp
    grid_size = interpob['grid_size']
    grid_size = tf.cast(grid_size, om.dtype)
    numpoints = interpob['numpoints']
    numpoints = tf.cast(numpoints, om.dtype)
    ndims = om.shape[1]

    # convert to normalized freq locs
    # the frequencies are originally in [-pi; pi]
    # we put them in [-grid_size/2; grid_size/2]
    pi = tf.constant(m.pi)
    tm = om * grid_size[None, :, None] / tf.cast(2 * pi, om.dtype)
    Jgen = []
    for i in range(ndims):
        # number of points to use for interpolation is numpoints
        Jgen.append(tf.range(numpoints[i]))
    # build an iterator for going over all J values
    Jgen = product(Jgen, im_rank)
    Jgen = tf.cast(Jgen, 'int64')
    # set up params if not using sparse mats
    params = {
        'dims': None,
        'table': interpob['table'],
        'numpoints': numpoints,
        'Jlist': Jgen,
        'table_oversamp': interpob['table_oversamp'],
    }

    x = tf.TensorArray(y.dtype, tf.shape(y)[0])
    y_shifted = tf.zeros(
        shape=(tf.shape(y)[1], tf.cast(tf.reduce_prod(grid_size), tf.int32)),
        dtype=y.dtype,
    )
    # run the table interpolator for each batch element
    # TODO: look into how to use tf.while_loop
    for b in tf.range(tf.shape(y)[0]):
        # phase for fftshift
        y_shifted = y[b] * tf.math.conj(tf.exp(1j * tf.cast(tf.linalg.matvec(tf.transpose(om[b]), n_shift), y[b].dtype))[None, ...])

        params['dims'] = tf.cast(grid_size, 'int64')

        x = x.write(b, run_interp_back(y_shifted, tm[b], params))

    x = x.stack()

    bsize = tf.shape(y)[0]
    ncoil = tf.shape(y)[1]
    out_size = tf.concat([[bsize, ncoil], tf.cast(grid_size, 'int64')], 0)

    x = tf.reshape(x, out_size)

    return x
