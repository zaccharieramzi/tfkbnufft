import tensorflow as tf
import numpy as np
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule


def test_adjoint_and_gradients():
    N = 20
    M = 20*5
    nufft_ob = KbNufftModule(im_size=(N, N), grid_size=(2*N, 2*N), norm='ortho', grad_traj=True)
    # Generate Trajectory
    ktraj = tf.Variable(tf.random.uniform((1, 2, M), minval=-1/2, maxval=1/2)*2*np.pi)
    # Have a random signal
    signal = tf.Variable(tf.cast(tf.random.uniform((1, 1, N, N)), tf.complex64))
    kdata = tf.Variable(kbnufft_forward(nufft_ob._extract_nufft_interpob())(signal, ktraj))

    with tf.GradientTape(persistent=True) as g:
        I_nufft = kbnufft_adjoint(nufft_ob._extract_nufft_interpob())(kdata, ktraj)[0][0]
        r = tf.cast(tf.reshape(tf.meshgrid(
            tf.linspace(-N/2, N/2-1, N),
            tf.linspace(-N/2, N/2-1, N),
            indexing='ij'
        ), (2, N * N)), tf.float32)
        A = tf.exp(2j * np.pi * tf.cast(tf.matmul(tf.transpose(ktraj[0]/2/np.pi), r), tf.complex64))/N/2
        I_ndft = tf.reshape(tf.matmul(tf.transpose(A), kdata[0][0][..., None]), (N, N))

    tf_test = tf.test.TestCase()
    # Test if the NUFFT and NDFT operation is same
    tf_test.assertAllClose(I_nufft, I_ndft, rtol=1e-1)

    # Test gradients with respect to kdata
    gradient_ndft_kdata = g.gradient(I_ndft, kdata)[0]
    gradient_nufft_kdata = g.gradient(I_nufft, kdata)[0]
    tf_test.assertAllClose(gradient_ndft_kdata, gradient_nufft_kdata, atol=1e-1)

    # Test gradients with respect to trajectory location
    gradient_ndft_traj = g.gradient(I_ndft, ktraj)[0]
    gradient_nufft_traj = g.gradient(I_nufft, ktraj)[0]
    tf_test.assertAllClose(gradient_ndft_traj, gradient_nufft_traj, atol=1e-1)
    # This is gradient of NDFT from matrix, will help in debug
    # gradient_from_matrix = 2*np.pi*1j*tf.matmul(tf.cast(r, tf.complex64), tf.transpose(A))*kdata[0][0]


def test_forward_and_gradients():
    N = 20
    M = 20*5
    nufft_ob = KbNufftModule(im_size=(N, N), grid_size=(2*N, 2*N), norm='ortho', grad_traj=True)
    # Generate Trajectory
    ktraj = tf.Variable(tf.random.uniform((1, 2, M), minval=-1/2, maxval=1/2)*2*np.pi)
    # Have a random signal
    signal = tf.Variable(tf.cast(tf.random.uniform((1, 1, N, N)), tf.complex64))

    with tf.GradientTape(persistent=True) as g:
        kdata_nufft = kbnufft_forward(nufft_ob._extract_nufft_interpob())(signal, ktraj)[0]
        r = tf.cast(tf.reshape(tf.meshgrid(
            tf.linspace(-N/2, N/2-1, N),
            tf.linspace(-N/2, N/2-1, N),
            indexing='ij'
        ), (2, N * N)), tf.float32)
        A = tf.exp(-2j * np.pi * tf.cast(tf.matmul(tf.transpose(ktraj[0])/2/np.pi, r), tf.complex64))/N/2
        kdata_ndft = tf.transpose(tf.matmul(A, tf.reshape(signal[0][0], (N*N, 1))))

    tf_test = tf.test.TestCase()
    # Test if the NUFFT and NDFT operation is same
    tf_test.assertAllClose(kdata_nufft, kdata_ndft, atol=1e-1)

    # Test gradients with respect to kdata
    gradient_ndft_kdata = g.gradient(kdata_ndft, signal)[0]
    gradient_nufft_kdata = g.gradient(kdata_nufft, signal)[0]
    tf_test.assertAllClose(gradient_ndft_kdata, gradient_nufft_kdata, atol=1)

    # Test gradients with respect to trajectory location
    gradient_ndft_traj = g.gradient(kdata_ndft, ktraj)[0]
    gradient_nufft_traj = g.gradient(kdata_nufft, ktraj)[0]
    tf_test.assertAllClose(gradient_ndft_traj, gradient_nufft_traj, atol=1)
    # This is gradient of NDFT from matrix, will help in debug
    # gradient_ndft_matrix = -2j * np.pi * tf.transpose(tf.matmul(A, tf.transpose(tf.cast(r, tf.complex64) * tf.reshape(signal[0][0], (N*N,)))))

