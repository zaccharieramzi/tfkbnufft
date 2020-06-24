import time

import numpy as np
from PIL import Image
from skimage.data import camera
import tensorflow as tf

from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule


def profile_tfkbnufft(
        image,
        ktraj,
        im_size,
        device,
    ):
    if device == 'CPU':
        num_nuffts = 20
    else:
        num_nuffts = 50
    print(f'Using {device}')
    device_name = f'/{device}:0'
    with tf.device(device_name):
        image = tf.constant(image)
        if device == 'GPU':
            image = tf.cast(image, tf.complex64)
        ktraj = tf.constant(ktraj)
        nufft_ob = KbNufftModule(im_size=im_size, grid_size=None, norm='ortho')
        forward_op = kbnufft_forward(nufft_ob._extract_nufft_interpob())
        adjoint_op = kbnufft_adjoint(nufft_ob._extract_nufft_interpob())

        # warm-up computation
        for _ in range(2):
            y = forward_op(image, ktraj)

        start_time = time.perf_counter()
        for _ in range(num_nuffts):
            y = forward_op(image, ktraj)
        end_time = time.perf_counter()
        avg_time = (end_time-start_time) / num_nuffts
        print('forward average time: {}'.format(avg_time))

        # warm-up computation
        for _ in range(2):
            x = adjoint_op(y, ktraj)

        # run the adjoint speed tests
        start_time = time.perf_counter()
        for _ in range(num_nuffts):
            x = adjoint_op(y, ktraj)
        end_time = time.perf_counter()
        avg_time = (end_time-start_time) / num_nuffts
        print('backward average time: {}'.format(avg_time))


def run_all_profiles():
    print('running profiler...')
    spokelength = 512
    nspokes = 405

    print('problem size (radial trajectory, 2-factor oversampling):')
    print('number of spokes: {}'.format(nspokes))
    print('spokelength: {}'.format(spokelength))

    # create an example to run on
    image = np.array(Image.fromarray(camera()).resize((256, 256)))
    image = image.astype(np.complex)
    im_size = image.shape

    image = image[None, None, ...]

    # create k-space trajectory
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

    ktraj = ktraj[None, ...]

    profile_tfkbnufft(image, ktraj, im_size, device='CPU')
    profile_tfkbnufft(image, ktraj, im_size, device='GPU')


if __name__ == '__main__':
    run_all_profiles()
