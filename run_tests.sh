#!/usr/bin/env bash
set -e
pip install torch==1.7 torchkbnufft==0.3.4 scikit-image pytest
# We test ndft_test.py separately as it causes some issues with tracing resulting in hangs.
python -m pytest tfkbnufft --ignore=tfkbnufft/tests/ndft_test.py
python -m pytest tfkbnufft/tests/ndft_test.py
