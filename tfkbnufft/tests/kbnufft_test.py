import numpy as np
import pytest
import tensorflow as tf

from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule


image_shape = (640, 400)
nspokes = 15
spokelength = image_shape[-1] * 2
kspace_shape = spokelength * nspokes

def ktraj_function():
    # radial trajectory creation
    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

    ky = np.transpose(ky)
    kx = np.transpose(kx)

    traj = np.stack((ky.flatten(), kx.flatten()), axis=0)
    traj = tf.convert_to_tensor(traj)[None, ...]
    return traj

@pytest.mark.parametrize('multiprocessing', [True, False])
def test_adjoint_gradient(multiprocessing):
    traj = ktraj_function()
    kspace = tf.zeros([1, 1, kspace_shape], dtype=tf.complex64)
    nufft_ob = KbNufftModule(
        im_size=(640, 400),
        grid_size=None,
        norm='ortho',
    )
    backward_op = kbnufft_adjoint(nufft_ob._extract_nufft_interpob(), multiprocessing)
    with tf.GradientTape() as tape:
        tape.watch(kspace)
        res = backward_op(kspace, traj)
    grad = tape.gradient(res, kspace)
    tf_test = tf.test.TestCase()
    tf_test.assertEqual(grad.shape, kspace.shape)

@pytest.mark.parametrize('multiprocessing', [True, False])
def test_forward_gradient(multiprocessing):
    traj = ktraj_function()
    image = tf.zeros([1, 1, *image_shape], dtype=tf.complex64)
    nufft_ob = KbNufftModule(
        im_size=(640, 400),
        grid_size=None,
        norm='ortho',
    )
    forward_op = kbnufft_forward(nufft_ob._extract_nufft_interpob(), multiprocessing)
    with tf.GradientTape() as tape:
        tape.watch(image)
        res = forward_op(image, traj)
    grad = tape.gradient(res, image)
    tf_test = tf.test.TestCase()
    tf_test.assertEqual(grad.shape, image.shape)
