# Image Classification Benchmark execution with MLCube

This is the README for executing the benchmark on MLCube (using Tensorflow2). The original README for using the TensorFlow2 model can be found [here](./README_tensorflow2.md). The README for using the TensorFlow1 model is [here](./README_tensorflow1.md).

## Project setup

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the RNN speech recognition workload
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/508/head:feature/mlcube_image_classification
git checkout feature/mlcube_image_classification && cd ./image_classification/mlcube
```

## Dataset

The [ImageNet](https://www.image-net.org/) needs to be downloaded manually.

## Tasks execution

```bash
# Download ImageNet dataset. Default path = /workspace/data
# To override it, use data_dir=DATA_DIR
mlcube run --task download
```

By default MLCube images use pull-type installation, so they should be available on docker hub. If not, try this:

```bash
mlcube run ... -Pdocker.build_strategy=auto
```

We are targeting pull-type installation, so MLCube images should be available on docker hub. If not, try this:

```bash
mlcube run ... -Pdocker.build_strategy=always
```