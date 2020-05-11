# TF KB-NUFFT

[GitHub](https://github.com/zaccharieramzi/tfkbnufft) | [![Build Status](https://travis-ci.com/zaccharieramzi/tfkbnufft.svg?branch=master)](https://travis-ci.com/zaccharieramzi/tfkbnufft)


Simple installation from pypi:
```
pip install tfkbnufft
```

## About

This package is a verly early-stage and modest adaptation to TensorFlow of the [torchkbnufft](https://github.com/mmuckley/torchkbnufft) package written by Matthew Muckley for PyTorch.
Please cite his work appropriately if you use this package.

## References

1. Fessler, J. A., & Sutton, B. P. (2003). Nonuniform fast Fourier transforms using min-max interpolation. *IEEE transactions on signal processing*, 51(2), 560-574.

2. Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005). Rapid gridding reconstruction with a minimal oversampling ratio. *IEEE transactions on medical imaging*, 24(6), 799-808.

3. Feichtinger, H. G., Gr, K., & Strohmer, T. (1995). Efficient numerical methods in non-uniform sampling theory. Numerische Mathematik, 69(4), 423-440.

## Citation

If you want to cite the package, you can use any of the following:

```bibtex
@conference{muckley:20:tah,
  author = {M. J. Muckley and R. Stern and T. Murrell and F. Knoll},
  title = {{TorchKbNufft}: A High-Level, Hardware-Agnostic Non-Uniform Fast Fourier Transform},
  booktitle = {ISMRM Workshop on Data Sampling \& Image Reconstruction},
  year = 2020
}

@misc{Muckley2019,
  author = {Muckley, M.J. et al.},
  title = {Torch KB-NUFFT},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mmuckley/torchkbnufft}}
}
```
