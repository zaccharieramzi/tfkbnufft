import numpy as np
import tensorflow as tf
from ..nufft.interp_functions import kbinterp, adjkbinterp


def calculate_radial_dcomp_tf(interpob, nufftob_forw, nufftob_back, ktraj, stacks=False):
    """Numerical density compensation estimation for a radial trajectory.

    Estimates the density compensation function numerically using a NUFFT
    operator (nufftob_forw and nufftob_back) and a k-space trajectory (ktraj).
    The function applies A'A1 (where A is the nufftob and 1 is a ones vector)
    and estimates the signal accumulation at the origin of k-space. It then
    returns a vector of density compensation values that are computed based on
    the distance from the k-space center and thresholded above the center
    density estimate. Then, a density-compensated image can be calculated by
    applying A'Wy, where W is a diagonal matrix with the density compensation
    values.

    This function uses a nufft hyper parameter dictionary, the associated nufft
    operators and k-space trajectory.

    Args:
        interpob (dict): the output of `KbNufftModule._extract_nufft_interpob`
            containing all the hyper-parameters for the nufft computation.
        nufftob_forw (fun)
        nufftob_back (fun)
        ktraj (tensor): The k-space trajectory in radians/voxel dimension (d, m).
            d is the number of spatial dimensions, and m is the length of the
            trajectory.
        stacks (bool): whether the trajectory is actually a stacks of radial
            for 3D imaging rather than a pure radial trajectory. Not tested.
            Defaults to False.

    Returns:
        tensor: The density compensation coefficients for ktraj of size (m).
    """
    # remove sensitivities if dealing with MriSenseNufft
    if not interpob['norm'] == 'ortho':
        norm_factor = tf.reduce_prod(interpob['grid_size'])
    else:
        norm_factor = 1

    # append 0s for batch, first coil
    im_size = interpob['im_size']
    if len(im_size) != 3  and stacks:
        raise ValueError('`stacks` argument can only be used for 3d data')
    image_loc = tf.concat([
        (0, 0,),
        im_size // 2,
    ], axis=0)


    # get the size of the test signal (add batch, coil)
    test_size = tf.concat([(1, 1,), im_size], axis=0)

    test_sig = tf.ones(test_size, dtype=tf.complex64)

    # get one dcomp for each batch
    # extract the signal amplitude increase from center of image
    query_point = tf.gather_nd(
        nufftob_back(
            nufftob_forw(
                test_sig,
                ktraj[None, :]
            ),
            ktraj[None, :]
        ),
        [image_loc],
    ) / norm_factor

    # use query point to get ramp intercept
    threshold_level = tf.cast(1 / query_point, ktraj.dtype)

    # compute the new dcomp for the batch in batch_ind
    pi = tf.constant(np.pi, dtype=ktraj.dtype)
    if stacks:
        ktraj_thresh = ktraj[0:2]
    else:
        ktraj_thresh = ktraj
    dcomp = tf.maximum(
        tf.sqrt(tf.reduce_sum(ktraj_thresh ** 2, axis=0)) / pi,
        threshold_level,
    )

    return dcomp


def calculate_density_compensator(interpob, nufftob_forw, nufftob_back, ktraj, num_iterations=10):
    """Numerical density compensation estimation for a any trajectory.

    Estimates the density compensation function numerically using a NUFFT
    interpolator operator and a k-space trajectory (ktraj).
    This function implements Pipe et al

    This function uses a nufft hyper parameter dictionary, the associated nufft
    operators and k-space trajectory.

    Args:
        interpob (dict): the output of `KbNufftModule._extract_nufft_interpob`
            containing all the hyper-parameters for the nufft computation.
        nufftob_forw (fun)
        nufftob_back (fun)
        ktraj (tensor): The k-space trajectory in radians/voxel dimension (d, m).
            d is the number of spatial dimensions, and m is the length of the
            trajectory.
        num_iterations (int): default 10
            number of iterations

    Returns:
        tensor: The density compensation coefficients for ktraj of size (m).
    """
    test_sig = tf.ones([1, 1, ktraj.shape[1]], dtype=tf.complex64)
    for i in range(num_iterations):
        test_sig = test_sig / tf.cast(tf.math.abs(kbinterp(
            adjkbinterp(test_sig, ktraj[None, :], interpob),
            ktraj[None, :],
            interpob
        )), 'complex64')
    im_size = interpob['im_size']
    test_size = tf.concat([(1, 1,), im_size], axis=0)
    test_im = tf.ones(test_size, dtype=tf.complex64)
    test_im_recon = nufftob_back(
        test_sig * nufftob_forw(
            test_im,
            ktraj[None, :]
        ),
        ktraj[None, :]
    )
    ratio = tf.reduce_mean(test_im_recon)
    test_sig = test_sig / tf.cast(ratio, test_sig.dtype)
    test_sig = test_sig[0, 0]
    return test_sig
