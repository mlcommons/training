#!/bin/bash

nvidia-docker build . --rm -f Dockerfile.gpu -t ds2-cuda9cudnn7:gpu
