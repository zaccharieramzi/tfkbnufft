"""Package info"""

__version__ = '0.2.3'
__author__ = 'Zaccharie Ramzi'
__author_email__ = 'zaccharie.ramzi@inria.fr'
__license__ = 'MIT'
__homepage__ = 'https://github.com/zaccharieramzi/tfkbnufft'
__docs__ = 'A robust, easy-to-deploy non-uniform Fast Fourier Transform in TensorFlow.'

try:
    # This variable is injected in the __builtins__ by the build
    # process.
    __TFKBNUFFT_SETUP__
except NameError:
    __TFKBNUFFT_SETUP__ = False

if __TFKBNUFFT_SETUP__:
    import sys
    sys.stderr.write('Partial import of during the build process.\n')
else:
    # from .kbinterp import KbInterpBack, KbInterpForw
    from .kbnufft import kbnufft_forward, kbnufft_adjoint
    # from .mrisensenufft import MriSenseNufft, AdjMriSenseNufft, ToepSenseNufft
    from .nufft import utils as nufft_utils

    __all__ = [
        # 'KbInterpForw',
        # 'KbInterpBack',
        'kbnufft_forward',
        'kbnufft_adjoint',
        # 'MriSenseNufft',
        # 'AdjMriSenseNufft'
    ]
