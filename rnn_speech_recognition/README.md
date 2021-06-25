Implementation

We'll be updating this section as we merge MLCube PRs and make new MLCube releases.

```Python
# Create Python environment 
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker runner from GitHub repository (normally, users will just run `pip install mlcube mlcube_docker`)
git clone https://github.com/mlcommons/mlcube && cd ./mlcube
cd ./mlcube && python setup.py bdist_wheel  && pip install --force-reinstall ./dist/mlcube-* && cd ..
cd ./runners/mlcube_docker && python setup.py bdist_wheel  && pip install --force-reinstall --no-deps ./dist/mlcube_docker-* && cd ../../..

# Fetch the SSD workload
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/491/head:feature/rnnt_mlcube && git checkout feature/rnnt_mlcube
cd ./rnn_speech_recognition/pytorch

# Build MLCube docker image. We'll find a better way of integrating existing workloads
# with MLCube, so that MLCube runs this by itself (it can actually do it now, but in order
# to enable this, we would have to introduce more changes to the SSD repo).
docker build --build-arg http_proxy="${http_proxy}" --build-arg https_proxy="${https_proxy}" . -t mlcommons/train_rnn_speech_recognition:0.0.1 -f Dockerfile.mlcube

# Show tasks implemented in this MLCube.
cd ../mlcube && python3 -m pip install tornado && mlcube describe

# Download SSD dataset (~20 GB, ~40 GB space required). Default paths = ./workspace/cache and ./workspace/data
# To override them, use --cache_dir=CACHE_DIR and --data_dir=DATA_DIR
mlcube run --task download_data --platform docker

# Download ResNet34 feature extractor. Default path = ./workspace/data
# To override, use: --data_dir=DATA_DIR
mlcube run --task download_model --platform docker

# Run benchmark. Default paths = ./workspace/data
# Parameters to override: --data_dir=DATA_DIR, --pretrained_backbone=PATH_TO_RESNET3_WEIGHTS, --parameters_file=PATH_TO_TRAINING_PARAMS
mlcube run --task train --platform docker
```