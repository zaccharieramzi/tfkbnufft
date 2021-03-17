import multiprocessing

import tensorflow as tf
from tensorflow.python.ops.signal.fft_ops import ifft2d, fft2d, fft, ifft

def tf_mp_ifft(kspace):
    k_shape_x = tf.shape(kspace)[-1]
    batched_kspace = tf.reshape(kspace, (-1, k_shape_x))
    batched_image = tf.map_fn(
        ifft,
        batched_kspace,
        parallel_iterations=multiprocessing.cpu_count(),
    )
    image = tf.reshape(batched_image, tf.shape(kspace))
    return image

def tf_mp_fft(kspace):
    k_shape_x = tf.shape(kspace)[-1]
    batched_kspace = tf.reshape(kspace, (-1, k_shape_x))
    batched_image = tf.map_fn(
        fft,
        batched_kspace,
        parallel_iterations=multiprocessing.cpu_count(),
    )
    image = tf.reshape(batched_image, tf.shape(kspace))
    return image

def tf_mp_ifft2d(kspace):
    k_shape_x = tf.shape(kspace)[-2]
    k_shape_y = tf.shape(kspace)[-1]
    batched_kspace = tf.reshape(kspace, (-1, k_shape_x, k_shape_y))
    batched_image = tf.map_fn(
        ifft2d,
        batched_kspace,
        parallel_iterations=multiprocessing.cpu_count(),
    )
    image = tf.reshape(batched_image, tf.shape(kspace))
    return image

def tf_mp_fft2d(image):
    shape_x = tf.shape(image)[-2]
    shape_y = tf.shape(image)[-1]
    batched_image = tf.reshape(image, (-1, shape_x, shape_y))
    batched_kspace = tf.map_fn(
        fft2d,
        batched_image,
        parallel_iterations=multiprocessing.cpu_count(),
    )
    kspace = tf.reshape(batched_kspace, tf.shape(image))
    return kspace

def tf_mp_ifft3d(kspace):
    image = tf_mp_fourier3d(kspace, trans_type='inv')
    return image

def tf_mp_fft3d(image):
    kspace = tf_mp_fourier3d(image, trans_type='forw')
    return kspace

def tf_mp_fourier3d(x, trans_type='inv'):
    fn_2d, fn_1d = (ifft2d, ifft) if trans_type == 'inv' else (fft2d, fft)
    n_slices = tf.shape(x)[0]
    n_coils = tf.shape(x)[1]
    shape_z = tf.shape(x)[-3]
    shape_x = tf.shape(x)[-2]
    shape_y = tf.shape(x)[-1]
    reshaped_x = tf.reshape(x, (-1, shape_x, shape_y))
    batched_incomplete_y = tf.map_fn(
        fn_2d,
        reshaped_x,
        parallel_iterations=multiprocessing.cpu_count(),
    )
    incomplete_y = tf.reshape(batched_incomplete_y, tf.shape(x))
    incomplete_y_reshaped = tf.transpose(incomplete_y, [0, 1, 3, 4, 2])
    batched_incomplete_y_reshaped = tf.reshape(incomplete_y_reshaped, (-1, shape_z))
    batched_y = tf.map_fn(
        fn_1d,
        batched_incomplete_y_reshaped,
        parallel_iterations=multiprocessing.cpu_count(),
    )
    y_reshaped = tf.reshape(batched_y, [n_slices, n_coils, shape_x, shape_y, shape_z])
    y = tf.transpose(y_reshaped, [0, 1, 4, 2, 3])
    return y

# Generate a fourier dictionary to simplify its use below.
# In the end we have the following list:
# fourier_dict[do_ifft][multiprocessing][rank of image - 1]
fourier_list = [
    [
        [
            tf.signal.fft,
            tf.signal.fft2d,
            tf.signal.fft3d,
        ],
        [
            tf_mp_fft,
            tf_mp_fft2d,
            tf_mp_fft3d,
        ]
    ],
    [
        [
            tf.signal.ifft,
            tf.signal.ifft2d,
            tf.signal.ifft3d,
        ],
        [
            tf_mp_ifft,
            tf_mp_ifft2d,
            tf_mp_ifft3d,
        ]
    ]
]


def scale_and_fft_on_image_volume(x, scaling_coef, grid_size, im_size, norm, im_rank=2, multiprocessing=False,
                                  do_ifft=False):
    """Applies the FFT and any relevant scaling factors to x.

    Args:
        x (tensor): The image to be FFT'd.
        scaling_coef (tensor): The NUFFT scaling coefficients to be multiplied
            prior to FFT.
        grid_size (tensor): The oversampled grid size.
        im_size (tensor): The image dimensions for x.
        norm (str): Type of normalization factor to use. If 'ortho', uses
            orthogonal FFT, otherwise, no normalization is applied.
        do_ifft (bool, optional, default False): When true, the IFFT is
            carried out on signal rather than FFT. This is needed for gradient.

    Returns:
        tensor: The oversampled FFT of x.
    """
    # zero pad for oversampled nufft
    # we don't need permutations since the fft in fourier is done on the
    # innermost dimensions and we are handling complex tensors
    pad_sizes = [
        (0, 0),  # batch dimension
        (0, 0),  # coil dimension
    ] + [
        (0, grid_size[0] - im_size[0]),  # nx
    ]
    if im_rank >= 2:
        pad_sizes += [(0, grid_size[1] - im_size[1])]
    if im_rank == 3:
        pad_sizes += [(0, grid_size[2] - im_size[2])]  # nz
    scaling_coef = tf.cast(scaling_coef, x.dtype)
    scaling_coef = scaling_coef[None, None, ...]
    # multiply by scaling coefs
    if do_ifft:
        x = x * tf.math.conj(scaling_coef)
    else:
        x = x * scaling_coef

    # zero pad and fft
    x = tf.pad(x, pad_sizes)
    x = fourier_list[do_ifft][multiprocessing][im_rank - 1](x)
    if norm == 'ortho':
        scaling_factor = tf.cast(tf.reduce_prod(grid_size), x.dtype)
        if do_ifft:
            x = x * tf.sqrt(scaling_factor)
        else:
            x = x / tf.sqrt(scaling_factor)

    return x


def ifft_and_scale_on_gridded_data(x, scaling_coef, grid_size, im_size, norm, im_rank=2, multiprocessing=False):
    """Applies the iFFT and any relevant scaling factors to x.

    Args:
        x (tensor): The gridded data to be iFFT'd.
        scaling_coef (tensor): The NUFFT scaling coefficients to be multiplied
            after iFFT.
        grid_size (tensor): The oversampled grid size.
        im_size (tensor): The image dimensions for x.
        norm (str): Type of normalization factor to use. If 'ortho', uses
            orthogonal iFFT, otherwise, no normalization is applied.

    Returns:
        tensor: The iFFT of x.
    """
    # we don't need permutations since the fft in fourier is done on the
    # innermost dimensions and we are handling complex tensors
    # do the inverse fft
    x = fourier_list[True][multiprocessing][im_rank - 1](x)
    im_size = tf.cast(im_size, tf.int32)
    # crop to output size
    x = x[:, :, :im_size[0]]
    if im_rank >=2:
        if im_rank == 3:
            x = x[..., :im_size[1], :im_size[2]]
        else:
            x = x[..., :im_size[1]]

    # scaling
    scaling_factor = tf.cast(tf.reduce_prod(grid_size), x.dtype)
    if norm == 'ortho':
        x = x * tf.sqrt(scaling_factor)
    else:
        x = x * scaling_factor

    # scaling coefficient multiply
    scaling_coef = tf.cast(scaling_coef, x.dtype)
    scaling_coef = scaling_coef[None, None, ...]

    x = x * tf.math.conj(scaling_coef)
    # this might be nice to try at some point more like an option rather
    # than a try except.
    # # try to broadcast multiply - batch over coil if not enough memory
    # raise_error = False
    # try:
    #     x = x * tf.math.conj(scaling_coef)
    # except RuntimeError as e:
    #     if 'out of memory' in str(e) and not raise_error:
    #         torch.cuda.empty_cache()
    #         for coilind in range(x.shape[1]):
    #             x[:, coilind, ...] = conj_complex_mult(
    #                 x[:, coilind:coilind + 1, ...], scaling_coef, dim=2)
    #         raise_error = True
    #     else:
    #         raise e
    # except BaseException:
    #     raise e
    #
    return x

# used for toep thing
# def fft_filter(x, kern, norm=None):
#     """FFT-based filtering on a 2-size oversampled grid.
#     """
#     x = x.clone()
#
#     im_size = torch.tensor(x.shape).to(torch.long)[3:]
#     grid_size = im_size * 2
#
#     # set up n-dimensional zero pad
#     pad_sizes = []
#     permute_dims = [0, 1]
#     inv_permute_dims = [0, 1, 2 + grid_size.shape[0]]
#     for i in range(grid_size.shape[0]):
#         pad_sizes.append(0)
#         pad_sizes.append(int(grid_size[-1 - i] - im_size[-1 - i]))
#         permute_dims.append(3 + i)
#         inv_permute_dims.append(2 + i)
#     permute_dims.append(2)
#     pad_sizes = tuple(pad_sizes)
#     permute_dims = tuple(permute_dims)
#     inv_permute_dims = tuple(inv_permute_dims)
#
#     # zero pad and fft
#     x = F.pad(x, pad_sizes)
#     x = x.permute(permute_dims)
#     x = torch.fft(x, grid_size.numel())
#     if norm == 'ortho':
#         x = x / torch.sqrt(torch.prod(grid_size.to(torch.double)))
#     x = x.permute(inv_permute_dims)
#
#     # apply the filter
#     x = complex_mult(x, kern, dim=2)
#
#     # inverse fft
#     x = x.permute(permute_dims)
#     x = torch.ifft(x, grid_size.numel())
#     x = x.permute(inv_permute_dims)
#
#     # crop to input size
#     crop_starts = tuple(np.array(x.shape).astype(np.int) * 0)
#     crop_ends = [x.shape[0], x.shape[1], x.shape[2]]
#     for dim in im_size:
#         crop_ends.append(int(dim))
#     x = x[tuple(map(slice, crop_starts, crop_ends))]
#
#     # scaling, assume user handled adjoint scaling with their kernel
#     if norm == 'ortho':
#         x = x / torch.sqrt(torch.prod(grid_size.to(torch.double)))
#
#     return x
