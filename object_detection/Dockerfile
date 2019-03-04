FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils=1.2.29ubuntu0.1 \
                       libglib2.0-0=2.48.2-0ubuntu4.1 \
                       libsm6=2:1.2.2-1 \
                       libxext6=2:1.3.3-1 \
                       libxrender-dev=1:0.9.9-0ubuntu1

RUN pip install ninja==1.8.2.post2 \
                yacs==0.1.5 \
                cython==0.29.5 \
                matplotlib==3.0.2 \
                opencv-python==4.0.0.21 \
                mlperf_compliance==0.0.10 \
                torchvision==0.2.2

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && git reset --hard ed842bffd41f6ff38707c4f0968d2cfd91088688 \
 && python setup.py build_ext install

# For information purposes only, these are the versions of the packages which we've successfully used:
# $ pip list
# Package              Version           Location
# -------------------- ----------------- -------------------------------------------------
# backcall             0.1.0
# certifi              2018.11.29
# cffi                 1.11.5
# cycler               0.10.0
# Cython               0.29.5
# decorator            4.3.2
# fairseq              0.6.0             /scratch/fairseq
# ipython              7.2.0
# ipython-genutils     0.2.0
# jedi                 0.13.2
# kiwisolver           1.0.1
# maskrcnn-benchmark   0.1               /scratch/mlperf/training/object_detection/pytorch
# matplotlib           3.0.2
# mkl-fft              1.0.10
# mkl-random           1.0.2
# mlperf-compliance    0.0.10
# ninja                1.8.2.post2
# numpy                1.16.1
# opencv-python        4.0.0.21
# parso                0.3.2
# pexpect              4.6.0
# pickleshare          0.7.5
# Pillow               5.4.1
# pip                  19.0.1
# prompt-toolkit       2.0.8
# ptyprocess           0.6.0
# pycocotools          2.0
# pycparser            2.19
# Pygments             2.3.1
# pyparsing            2.3.1
# python-dateutil      2.8.0
# pytorch-quantization 0.2.1
# PyYAML               3.13
# setuptools           40.8.0
# six                  1.12.0
# torch                1.0.0.dev20190225
# torchvision          0.2.1
# tqdm                 4.31.1
# traitlets            4.3.2
# wcwidth              0.1.7
# wheel                0.32.3
# yacs                 0.1.5
