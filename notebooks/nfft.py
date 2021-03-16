import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule


N = 1024
M = 512
nufft_ob = KbNufftModule(im_size=(N, ), grid_size=(2*N, ), norm='ortho')
ktraj = tf.Variable(tf.random.uniform((1, 1, M), minval=-1/2, maxval=1/2))
ktraj_seq = tf.Variable(tf.cast(np.linspace(-1/2, 1/2, N, endpoint=False)[None, None, :], tf.float32))
signal = tf.Variable(tf.cast(tf.random.uniform((1, 1, N)), tf.complex64))
signal_spectrum = tf.signal.fftshift(tf.signal.fft(tf.signal.ifftshift(signal[0][0]))) / N

with tf.GradientTape(persistent=True) as g:
    kdata_nufft = kbnufft_forward(nufft_ob._extract_nufft_interpob())(signal, ktraj*np.pi*2)
    r = tf.cast(tf.linspace(0, N-1, N), tf.float32)
    A = tf.exp(-2j * np.pi * tf.cast(ktraj[0][0][..., None] * r[None, :], tf.complex64))/N
    kdata_ndft = tf.matmul(A, tf.transpose(signal[0]))

grad1 = g.gradient(kdata_nufft, ktraj)
grad2 = g.gradient(kdata_ndft, ktraj)
grad3 = -2j * np.pi * tf.matmul(A, tf.transpose(signal[0] * tf.cast(r, tf.complex64)))
grad_nufft_mine =-2j * np.pi * kbnufft_forward(nufft_ob._extract_nufft_interpob())(signal * tf.cast(r, tf.complex64), ktraj*np.pi*2)