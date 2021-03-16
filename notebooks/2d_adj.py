import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule
from mri.operators import NonCartesianFFT
tf.config.run_functions_eagerly(True)
tf.random.set_seed(0)
N = 20
M = 20*5
nufft_ob = KbNufftModule(im_size=(N, N), grid_size=(2*N, 2*N), norm='ortho')
ktraj = tf.Variable(tf.random.uniform((1, 2, M), minval=-1/2, maxval=1/2))
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
kdata = tf.Variable(kbnufft_forward(nufft_ob._extract_nufft_interpob())(signal, ktraj*np.pi*2))

with tf.GradientTape(persistent=True) as g:
    I_nufft = kbnufft_adjoint(nufft_ob._extract_nufft_interpob())(kdata, ktraj*np.pi*2)[0][0]
    r = tf.cast(tf.reshape(tf.meshgrid(
        np.linspace(-N/2, N/2, N, endpoint=False),
        np.linspace(-N/2, N/2, N, endpoint=False),
        indexing='ij'
    ), (2, N*N)), tf.float32)
    A = tf.exp(2j * np.pi * tf.cast(tf.matmul(tf.transpose(ktraj[0]), r), tf.complex64))/N/2
    I_ndft = tf.reshape(tf.matmul(tf.transpose(A), kdata[0][0][..., None]), (N, N))

grad2 = g.gradient(I_ndft, ktraj)[0]
grad3 = 2*np.pi*1j*tf.matmul(tf.cast(r, tf.complex64), tf.transpose(A))*kdata[0][0]
grad4 = g.gradient(I_nufft, ktraj)[0]
C = kbnufft_adjoint(nufft_ob._extract_nufft_interpob())(kdata, ktraj*np.pi*2) * tf.reshape(tf.cast(r, tf.complex64), (1, 2, 20, 20))
grad4 = tf.reshape(
    2j * np.pi * kbnufft_adjoint(nufft_ob._extract_nufft_interpob())(
        kdata * tf.cast(r, tf.complex64),
        ktraj*np.pi*2
    ),
    (2, N*N)
)
grad4