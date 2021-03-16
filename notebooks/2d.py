import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule
from mri.operators import NonCartesianFFT
tf.config.run_functions_eagerly(True)

N = 20
M = 20*5
nufft_ob = KbNufftModule(im_size=(N, N), grid_size=(2*N, 2*N), norm='ortho')
ktraj = tf.Variable(tf.random.uniform((1, 2, M), minval=-1/2, maxval=1/2)*2*np.pi)
ktraj_seq = tf.Variable(tf.cast(
    np.reshape(np.meshgrid(
        np.linspace(-1/2, 1/2, N, endpoint=False),
        np.linspace(-1/2, 1/2, N, endpoint=False),
        indexing='ij'
    ), (2, N*N)
    )[None, :],
    tf.float32
))
signal = tf.Variable(tf.cast(tf.random.uniform((1, 1, N, N)), tf.complex64))
with tf.GradientTape(persistent=True) as g:
    kdata_nufft = kbnufft_forward(nufft_ob._extract_nufft_interpob())(signal, ktraj)
    r = tf.cast(tf.reshape(tf.meshgrid(
        np.linspace(-N/2, N/2, N, endpoint=False),
        np.linspace(-N/2, N/2, N, endpoint=False),
        indexing='ij'
    ), (2, N*N)), tf.float32)
    A = tf.exp(-2j * np.pi * tf.cast(tf.matmul(tf.transpose(ktraj[0])/2/np.pi, r), tf.complex64))/N/2
    kdata_ndft = tf.matmul(A, tf.reshape(signal[0][0], (N*N, 1)))

grad2 = g.gradient(kdata_ndft, ktraj)[0]
grad3 = -2j * np.pi * tf.transpose(tf.matmul(A, tf.transpose(tf.cast(r, tf.complex64) * tf.reshape(signal[0][0], (N*N,)))))
grad4 = g.gradient(kdata_nufft, ktraj)[0]