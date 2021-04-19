import warnings

import numpy as np
import tensorflow as tf

# from .functional.kbnufft import AdjKbNufftFunction, KbNufftFunction
                                 # ToepNufftFunction)
from .kbmodule import KbModule
from .nufft.fft_functions import scale_and_fft_on_image_volume, ifft_and_scale_on_gridded_data
from .nufft.interp_functions import kbinterp, adjkbinterp
from .nufft.utils import build_spmatrix, build_table, compute_scaling_coefs
from .utils.itertools import cartesian_product



class KbNufftModule(KbModule):
    """Parent class for KbNufft classes.

    This implementation collects all init functions into one place.

    Args:
        im_size (int or tuple of ints): Size of base image.
        grid_size (int or tuple of ints, default=2*im_size): Size of the grid
            to interpolate from.
        numpoints (int or tuple of ints, default=6): Number of points to use
            for interpolation in each dimension. Default is six points in each
            direction.
        n_shift (int or tuple of ints, default=im_size//2): Number of points to
            shift for fftshifts.
        table_oversamp (int, default=2^10): Table oversampling factor.
        kbwidth (double, default=2.34): Kaiser-Bessel width parameter.
        order (double, default=0): Order of Kaiser-Bessel kernel.
        norm (str, default='None'): Normalization for FFT. Default uses no
            normalization. Use 'ortho' to use orthogonal FFTs and preserve
            energy.
    """

    def __init__(self, im_size, grid_size=None, numpoints=6, n_shift=None,
                 table_oversamp=2**10, kbwidth=2.34, order=0, norm='None',
                 coil_broadcast=False, matadj=False, grad_traj=False):
        super(KbNufftModule, self).__init__()

        self.im_size = im_size
        self.im_rank = len(im_size)
        self.grad_traj = grad_traj
        if self.grad_traj:
            warnings.warn('The gradient w.r.t trajectory is Experimental and WIP. '
                          'Please use with caution')
        if grid_size is None:
            self.grid_size = tuple(np.array(self.im_size) * 2)
        else:
            self.grid_size = grid_size
        if n_shift is None:
            self.n_shift = tuple(np.array(self.im_size) // 2)
        else:
            self.n_shift = n_shift
        if isinstance(numpoints, int):
            self.numpoints = (numpoints,) * len(self.grid_size)
        else:
            self.numpoints = numpoints
        self.alpha = tuple(np.array(kbwidth) * np.array(self.numpoints))
        if isinstance(order, int) or isinstance(order, float):
            self.order = (order,) * len(self.grid_size)
        else:
            self.order = order
        if isinstance(table_oversamp, float) or isinstance(table_oversamp, int):
            self.table_oversamp = (table_oversamp,) * len(self.grid_size)
        else:
            self.table_oversamp = table_oversamp

        # dimension checking
        assert len(self.grid_size) == len(self.im_size)
        assert len(self.n_shift) == len(self.im_size)
        assert len(self.numpoints) == len(self.im_size)
        assert len(self.alpha) == len(self.im_size)
        assert len(self.order) == len(self.im_size)
        assert len(self.table_oversamp) == len(self.im_size)

        table = build_table(
            numpoints=self.numpoints,
            table_oversamp=self.table_oversamp,
            grid_size=self.grid_size,
            im_size=self.im_size,
            ndims=len(self.im_size),
            order=self.order,
            alpha=self.alpha
        )
        self.table = table
        assert len(self.table) == len(self.im_size)

        scaling_coef = compute_scaling_coefs(
            im_size=self.im_size,
            grid_size=self.grid_size,
            numpoints=self.numpoints,
            alpha=self.alpha,
            order=self.order
        )
        self.scaling_coef = scaling_coef
        self.norm = norm
        self.coil_broadcast = coil_broadcast
        self.matadj = matadj

        if coil_broadcast == True:
            warnings.warn(
                'coil_broadcast will be deprecated in a future release',
                DeprecationWarning)
        if matadj == True:
            warnings.warn(
                'matadj will be deprecated in a future release',
                DeprecationWarning)

        self.scaling_coef_tensor = tf.convert_to_tensor(self.scaling_coef)
        self.table_tensors = []
        for item in self.table:
            self.table_tensors.append(tf.convert_to_tensor(item))
        # register buffer is not necessary in tf, you just have the variable in
        # your class, point.
        self.n_shift_tensor = tf.convert_to_tensor(np.array(self.n_shift, dtype=np.int64))
        self.grid_size_tensor = tf.convert_to_tensor(np.array(self.grid_size, dtype=np.int64))
        self.im_size_tensor = tf.convert_to_tensor(np.array(self.im_size, dtype=np.int64))
        self.numpoints_tensor = tf.convert_to_tensor(np.array(self.numpoints, dtype=np.double))
        self.table_oversamp_tensor = tf.convert_to_tensor(np.array(self.table_oversamp, dtype=np.double))

    def _extract_nufft_interpob(self):
        """Extracts interpolation object from self.

        Returns:
            dict: An interpolation object for the NUFFT operation.
        """
        interpob = dict()
        interpob['scaling_coef'] = self.scaling_coef_tensor
        interpob['table'] = self.table_tensors
        interpob['n_shift'] = self.n_shift_tensor
        interpob['grid_size'] = self.grid_size_tensor
        interpob['im_size'] = self.im_size_tensor
        interpob['im_rank'] = self.im_rank
        interpob['numpoints'] = self.numpoints_tensor
        interpob['table_oversamp'] = self.table_oversamp_tensor
        interpob['norm'] = self.norm
        interpob['coil_broadcast'] = self.coil_broadcast
        interpob['matadj'] = self.matadj
        interpob['grad_traj'] = self.grad_traj
        Jgen = []
        for i in range(self.im_rank):
            # number of points to use for interpolation is numpoints
            Jgen.append(np.arange(self.numpoints[i]))
        Jgen = cartesian_product(Jgen)
        interpob['Jlist'] = Jgen.astype('int64')

        return interpob

def kbnufft_forward(interpob, multiprocessing=False):
    @tf.function(experimental_relax_shapes=True)
    @tf.custom_gradient
    def kbnufft_forward_for_interpob(x, om):
        """Apply FFT and interpolate from gridded data to scattered data.

        Inputs are assumed to be batch/chans x coil x image dims.
        Om should be nbatch x ndims x klength.

        Args:
            x (tensor): The original imagel.
            om (tensor, optional): A new set of omega coordinates at which to
                calculate the signal in radians/voxel.

        Returns:
            tensor: x computed at off-grid locations in om.
        """
        # this is with registered gradient, I would like to try without
        # y = KbNufftFunction.apply(x, om, interpob, interp_mats)
        # extract interpolation params
        scaling_coef = interpob['scaling_coef']
        grid_size = interpob['grid_size']
        im_size = interpob['im_size']
        norm = interpob['norm']
        grad_traj = interpob['grad_traj']
        im_rank = interpob.get('im_rank', 2)

        fft_x = scale_and_fft_on_image_volume(
            x, scaling_coef, grid_size, im_size, norm, im_rank=im_rank, multiprocessing=multiprocessing)

        y = kbinterp(fft_x, om, interpob)

        def grad(dy):
            # Gradients with respect to image
            grid_dy = adjkbinterp(dy, om, interpob)
            ifft_dy = ifft_and_scale_on_gridded_data(
                grid_dy, scaling_coef, grid_size, im_size, norm, im_rank=im_rank)
            if grad_traj:
                # Gradients with respect to trajectory locations
                r = [tf.linspace(-im_size[i]/2, im_size[i]/2-1, im_size[i]) for i in range(im_rank)]
                grid_r = tf.cast(tf.meshgrid(*r, indexing='ij'), x.dtype)[None, ...]
                fft_dx_dom = scale_and_fft_on_image_volume(
                x * grid_r, scaling_coef, grid_size, im_size, norm, im_rank=im_rank)
                dy_dom = tf.cast(-1j * tf.math.conj(dy) * kbinterp(fft_dx_dom, om, interpob), tf.float32)
            else:
                dy_dom = None
            return ifft_dy, dy_dom

        return y, grad
    return kbnufft_forward_for_interpob

def kbnufft_adjoint(interpob, multiprocessing=False):
    @tf.function(experimental_relax_shapes=True)
    @tf.custom_gradient
    def kbnufft_adjoint_for_interpob(y, om):
        """Interpolate from scattered data to gridded data and then iFFT.

        Inputs are assumed to be batch/chans x coil x kspace
        length. Om should be nbatch x ndims x klength.

        Args:
            y (tensor): The off-grid signal.
            om (tensor, optional): The off-grid coordinates in radians/voxel.

        Returns:
            tensor: The image after adjoint NUFFT.
        """
        grid_y = adjkbinterp(y, om, interpob)
        scaling_coef = interpob['scaling_coef']
        grid_size = interpob['grid_size']
        im_size = interpob['im_size']
        norm = interpob['norm']
        grad_traj = interpob['grad_traj']
        im_rank = interpob.get('im_rank', 2)
        ifft_y = ifft_and_scale_on_gridded_data(
            grid_y, scaling_coef, grid_size, im_size, norm, im_rank=im_rank, multiprocessing=multiprocessing)

        def grad(dx):
            # Gradients with respect to off grid signal
            fft_dx = scale_and_fft_on_image_volume(
                dx, scaling_coef, grid_size, im_size, norm, im_rank=im_rank)
            dx_dy = kbinterp(fft_dx, om, interpob)
            if grad_traj:
                # Gradients with respect to trajectory locations
                r = [tf.linspace(-im_size[i]/2, im_size[i]/2-1, im_size[i]) for i in range(im_rank)]
                grid_r = tf.cast(tf.meshgrid(*r, indexing='ij'), dx.dtype)[None, ...]
                ifft_dxr = scale_and_fft_on_image_volume(
                    tf.math.conj(dx) * grid_r, scaling_coef, grid_size, im_size, norm, im_rank=im_rank, do_ifft=True)
                dx_dom = tf.cast(1j * y * kbinterp(ifft_dxr, om, interpob, conj=True), om.dtype)
            else:
                dx_dom = None
            return dx_dy, dx_dom
        return ifft_y, grad
    return kbnufft_adjoint_for_interpob


# class ToepNufft(KbModule):
#     """Forward/backward NUFFT with Toeplitz embedding.
#
#     This module applies Tx, where T is a matrix such that T = A'A, where A is
#     a NUFFT matrix. Using Toeplitz embedding, this module computes the A'A
#     operation without interpolations, which is extremely fast.
#
#     The module is intended to be used in combination with an fft kernel
#     computed to be the frequency response of an embedded Toeplitz matrix. The
#     kernel is calculated offline via
#
#     torchkbnufft.nufft.toep_functions.calc_toep_kernel
#
#     The corresponding kernel is then passed to this module in its forward
#     forward operation, which applies a (zero-padded) fft filter using the
#     kernel.
#     """
#
#     def __init__(self):
#         super(ToepNufft, self).__init__()
#
#     def forward(self, x, kern, norm=None):
#         """Toeplitz NUFFT forward function.
#
#         Args:
#             x (tensor): The image (or images) to apply the forward/backward
#                 Toeplitz-embedded NUFFT to.
#             kern (tensor): The filter response taking into account Toeplitz
#                 embedding.
#             norm (str, default=None): Use 'ortho' if kern was designed to use
#                 orthogonal FFTs.
#
#         Returns:
#             tensor: x after applying the Toeplitz NUFFT.
#         """
#         x = ToepNufftFunction.apply(x, kern, norm)
#
#         return x
