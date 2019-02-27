#!/bin/bash

# Installs object_detection module

pushd pytorch

python setup.py build develop --user

popd
