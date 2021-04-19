import pytest
import tensorflow as tf
import numpy as np
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule


def get_fourier_matrix(ktraj, im_size, im_rank, do_ifft=False):
    r = [tf.linspace(-im_size[i]/2, im_size[i]/2-1, im_size[i]) for i in range(im_rank)]
    grid_r =tf.cast(tf.reshape(tf.meshgrid(*r ,indexing='ij'), (im_rank, tf.reduce_prod(im_size))), tf.float32)
    traj_grid = tf.cast(tf.matmul(tf.transpose(ktraj[0]), grid_r), tf.complex64)
    if do_ifft:
        A = tf.exp(1j * traj_grid)
    else:
        A = tf.exp(-1j * traj_grid)
    A = A / (np.sqrt(tf.reduce_prod(im_size)) * np.power(np.sqrt(2), im_rank))
    return A


@pytest.mark.parametrize('im_size', [(10, 10)])
def test_adjoint_and_gradients(im_size):
    tf.random.set_seed(0)
    grid_size = tuple(np.array(im_size)*2)
    im_rank = len(im_size)
    M = im_size[0] * 2**im_rank
    nufft_ob = KbNufftModule(im_size=im_size, grid_size=grid_size, norm='ortho', grad_traj=True)
    # Generate Trajectory
    ktraj_ori = tf.Variable(tf.random.uniform((1, im_rank, M), minval=-1/2, maxval=1/2)*2*np.pi)
    # Have a random signal
    signal = tf.Variable(tf.cast(tf.random.uniform((1, 1, *im_size)), tf.complex64))
    kdata = tf.Variable(kbnufft_forward(nufft_ob._extract_nufft_interpob())(signal, ktraj_ori))
    Idata = tf.Variable(kbnufft_adjoint(nufft_ob._extract_nufft_interpob())(kdata, ktraj_ori))
    ktraj_noise = np.copy(ktraj_ori)
    ktraj_noise += 0.01 * tf.Variable(tf.random.uniform((1, im_rank, M), minval=-1/2, maxval=1/2)*2*np.pi)
    ktraj = tf.Variable(ktraj_noise)
    with tf.GradientTape(persistent=True) as g:
        I_nufft = kbnufft_adjoint(nufft_ob._extract_nufft_interpob())(kdata, ktraj)[0][0]
        A = get_fourier_matrix(ktraj, im_size, im_rank, do_ifft=True)
        I_ndft = tf.reshape(tf.matmul(tf.transpose(A), kdata[0][0][..., None]), im_size)
        loss_nufft = tf.math.reduce_mean(tf.abs(Idata - I_nufft)**2)
        loss_ndft = tf.math.reduce_mean(tf.abs(Idata - I_ndft)**2)

    tf_test = tf.test.TestCase()
    # Test if the NUFFT and NDFT operation is same
    tf_test.assertAllClose(I_nufft, I_ndft, atol=1e-3)

    # Test gradients with respect to kdata
    gradient_ndft_kdata = g.gradient(I_ndft, kdata)[0]
    gradient_nufft_kdata = g.gradient(I_nufft, kdata)[0]
    tf_test.assertAllClose(gradient_ndft_kdata, gradient_nufft_kdata, atol=5e-3)

    # Test gradients with respect to trajectory location
    gradient_ndft_traj = g.gradient(I_ndft, ktraj)[0]
    gradient_nufft_traj = g.gradient(I_nufft, ktraj)[0]
    tf_test.assertAllClose(gradient_ndft_traj, gradient_nufft_traj, atol=5e-3)

    # Test gradients in chain rule with respect to ktraj
    gradient_ndft_loss = g.gradient(loss_ndft, ktraj)[0]
    gradient_nufft_loss = g.gradient(loss_nufft, ktraj)[0]
    tf_test.assertAllClose(gradient_ndft_loss, gradient_nufft_loss, atol=5e-4)

    # This is gradient of NDFT from matrix, will help in debug
    # gradient_from_matrix = 2*np.pi*1j*tf.matmul(tf.cast(r, tf.complex64), tf.transpose(A))*kdata[0][0]


@pytest.mark.parametrize('im_size', [(10, 10)])
def test_forward_and_gradients(im_size):
    tf.random.set_seed(0)
    grid_size = tuple(np.array(im_size)*2)
    im_rank = len(im_size)
    M = im_size[0] * 2**im_rank
    nufft_ob = KbNufftModule(im_size=im_size, grid_size=grid_size, norm='ortho', grad_traj=True)
    # Generate Trajectory
    ktraj_ori = tf.Variable(tf.random.uniform((1, im_rank, M), minval=-1/2, maxval=1/2)*2*np.pi)
    # Have a random signal
    signal = tf.Variable(tf.cast(tf.random.uniform((1, 1, *im_size)), tf.complex64))
    kdata = kbnufft_forward(nufft_ob._extract_nufft_interpob())(signal, ktraj_ori)[0]
    ktraj_noise = np.copy(ktraj_ori)
    ktraj_noise += 0.01 * tf.Variable(tf.random.uniform((1, im_rank, M), minval=-1/2, maxval=1/2)*2*np.pi)
    ktraj = tf.Variable(ktraj_noise)
    with tf.GradientTape(persistent=True) as g:
        kdata_nufft = kbnufft_forward(nufft_ob._extract_nufft_interpob())(signal, ktraj)[0]
        A = get_fourier_matrix(ktraj, im_size, im_rank, do_ifft=False)
        kdata_ndft = tf.transpose(tf.matmul(A, tf.reshape(signal[0][0], (tf.reduce_prod(im_size), 1))))
        loss_nufft = tf.math.reduce_mean(tf.abs(kdata - kdata_nufft)**2)
        loss_ndft = tf.math.reduce_mean(tf.abs(kdata - kdata_ndft)**2)

    tf_test = tf.test.TestCase()
    # Test if the NUFFT and NDFT operation is same
    tf_test.assertAllClose(kdata_nufft, kdata_ndft, atol=1e-3)

    # Test gradients with respect to kdata
    gradient_ndft_kdata = g.gradient(kdata_ndft, signal)[0]
    gradient_nufft_kdata = g.gradient(kdata_nufft, signal)[0]
    tf_test.assertAllClose(gradient_ndft_kdata, gradient_nufft_kdata, atol=5e-3)

    # Test gradients with respect to trajectory location
    gradient_ndft_traj = g.gradient(kdata_ndft, ktraj)[0]
    gradient_nufft_traj = g.gradient(kdata_nufft, ktraj)[0]
    tf_test.assertAllClose(gradient_ndft_traj, gradient_nufft_traj, atol=5e-3)

    # Test gradients in chain rule with respect to ktraj
    gradient_ndft_loss = g.gradient(loss_ndft, ktraj)[0]
    gradient_nufft_loss = g.gradient(loss_nufft, ktraj)[0]
    tf_test.assertAllClose(gradient_ndft_loss, gradient_nufft_loss, atol=5e-4)
    # This is gradient of NDFT from matrix, will help in debug
    # gradient_ndft_matrix = -1j * tf.transpose(tf.matmul(A, tf.transpose(tf.cast(grid_r, tf.complex64) * signal[0][0])))
