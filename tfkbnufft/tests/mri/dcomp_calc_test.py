import numpy as np
import tensorflow as tf

from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft.mri.dcomp_calc import calculate_radial_dcomp_tf


def setup():
    spokelength = 400
    grid_size = (spokelength, spokelength)
    nspokes = 10

    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

    ky = np.transpose(ky)
    kx = np.transpose(kx)

    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)
    nufft_ob = KbNufftModule(im_size=(200, 200), grid_size=grid_size, norm='ortho')
    return ktraj, nufft_ob

def test_calculate_radial_dcomp_tf():
    ktraj, nufft_ob = setup()
    interpob = nufft_ob._extract_nufft_interpob()
    nufftob_forw = kbnufft_forward(interpob)
    nufftob_back = kbnufft_adjoint(interpob)
    ktraj = tf.convert_to_tensor(ktraj)
    dcomp = calculate_radial_dcomp_tf(interpob, nufftob_forw, nufftob_back, ktraj)
