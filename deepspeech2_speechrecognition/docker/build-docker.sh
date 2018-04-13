#!/bin/bash

nvidia-docker build . --rm -f Dockerfile.gpu -t ds2-cuda8cudnn7:gpu
