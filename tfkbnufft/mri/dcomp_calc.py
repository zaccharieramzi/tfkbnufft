import numpy as np
import tensorflow as tf

def calculate_radial_dcomp_tf(interpob, nufftob_forw, nufftob_back, ktraj):
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

    Returns:
        tensor: The density compensation coefficients for ktraj of size (m).
    """
    # remove sensitivities if dealing with MriSenseNufft
    if not interpob['norm'] == 'ortho':
        norm_factor = tf.reduce_prod(interpob['grid_size'])
    else:
        norm_factor = 1

    # append 0s for batch, first coil, real part
    image_loc = tf.concat([
        (0, 0,),
        interpob['im_size'] // 2,
    ], axis=0)

    # get the size of the test signal (add batch, coil, real/imag dim)
    test_size = tf.concat([(1, 1,), interpob['im_size']], axis=0)

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
    dcomp = tf.maximum(
        tf.sqrt(tf.reduce_sum(ktraj[-2:, ...] ** 2, axis=0)) / pi,
        threshold_level,
    )

    return dcomp
