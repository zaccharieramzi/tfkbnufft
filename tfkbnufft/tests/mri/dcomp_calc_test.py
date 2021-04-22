import numpy as np
import tensorflow as tf
import torch

from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft.mri.dcomp_calc import calculate_radial_dcomp_tf, \
    calculate_density_compensator
from torchkbnufft import KbNufft, AdjKbNufft
from torchkbnufft.mri.dcomp_calc import calculate_radial_dcomp_pytorch


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
    im_size = (200, 200)
    nufft_ob = KbNufftModule(im_size=im_size, grid_size=grid_size, norm='ortho')
    torch_forward = KbNufft(im_size=im_size, grid_size=grid_size, norm='ortho')
    torch_backward = AdjKbNufft(im_size=im_size, grid_size=grid_size, norm='ortho')
    return ktraj, nufft_ob, torch_forward, torch_backward

def test_calculate_radial_dcomp_tf():
    ktraj, nufft_ob, torch_forward, torch_backward = setup()
    interpob = nufft_ob._extract_nufft_interpob()
    tf_nufftob_forw = kbnufft_forward(interpob)
    tf_nufftob_back = kbnufft_adjoint(interpob)
    tf_ktraj = tf.convert_to_tensor(ktraj)
    torch_ktraj = torch.tensor(ktraj).unsqueeze(0)
    tf_dcomp = calculate_radial_dcomp_tf(interpob, tf_nufftob_forw, tf_nufftob_back, tf_ktraj)
    torch_dcomp = calculate_radial_dcomp_pytorch(torch_forward, torch_backward, torch_ktraj)
    np.testing.assert_allclose(
        tf_dcomp.numpy(),
        torch_dcomp[0].numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

def test_density_compensators_tf():
    # This is a simple test to ensure that the code works only!
    # We still dont have a method to test if the results are correct
    ktraj, nufft_ob, torch_forward, torch_backward = setup()
    interpob = nufft_ob._extract_nufft_interpob()
    tf_ktraj = tf.convert_to_tensor(ktraj)
    nufftob_back = kbnufft_adjoint(interpob)
    nufftob_forw = kbnufft_forward(interpob)
    tf_dcomp = calculate_density_compensator(interpob, nufftob_forw, nufftob_back, tf_ktraj, zero_grad=False)
    tf_dcomp_no_grad = calculate_density_compensator(interpob, nufftob_forw, nufftob_back, tf_ktraj, zero_grad=True)
