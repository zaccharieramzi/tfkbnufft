#!/usr/bin/bash

rm -f -r build/*
rm -f -r dist/*

pip install --upgrade setuptools wheel

python setup.py sdist bdist_wheel
