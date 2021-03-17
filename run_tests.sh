#!/usr/bin/env bash
pip install torch==1.7 torchkbnufft==0.3.4 scikit-image pytest
python -m pytest tfkbnufft
