# MLCube for Single Stage Detector
MLCube™ GitHub [repository](https://github.com/mlcommons/mlcube). MLCube™ [wiki](https://mlcommons.github.io/mlcube/).

Clone project from GitHub.
```shell
git clone https://github.com/mlcommons/training
cd ./training/single_stage_detector
```

Create python virtual environment.
```shell
virtualenv -p python3 ./env
source ./env/bin/activate
pip install mlcube mlcube-docker
```

Build docker image and, optionally, push to a docker hub (this should be done by developers). If this is done, users
then skip this step, and MLCube will pull this docker image from the hub.
```shell
docker build --build-arg http_proxy="${http_proxy}" --build-arg https_proxy="${https_proxy}" . -t mlcommons/train_ssd:0.0.1 -f Dockerfile.mlcube
```

Go to mlcube directory and study what tasks MLCube implements.
```shell
cd ./mlcube
mlcube describe
```

Download COCO2017 dataset. The dataset consists of three directories - `train2017`, `val2017` and `annotations`. If you already have this data,
skip this step. Parameters to override:
- `--cache_dir=CACHE_DIR` Cache directory to store downloaded data (~20 GB required). Default is `${WORKSPACE}/cache`.
- `--data_dir=DATA_DIR` Dataset directory (~ 20 GB required). Default is `${WORKSPACE}/data`.

```shell
mlcube run --task download_data --platform docker
```

Download ResNet34 weights. Skip this step if you already have `resnet34-333f7ec4.pth` file. Parameters to override:
- `--model_dir=MODEL_DIR` Directory to store downloaded weights. Default is `${WORKSPACE}/data`.

```shell
mlcube run --task download_model --platform docker
```

Train SSD. Parameters to override:
- `--data_dir=DATA_DIR` Dataset directory. Default is `${WORKSPACE}/data`.
- `--pretrained_backbone=PATH_TO_FiLE` Full path to `resnet34-333f7ec4.pth` or other file.
  Default is `${WORKSPACE}/data/resnet34-333f7ec4.pth`.
- `--parameters_file=PATH_TO_FILE` YAML file with parameters. Default is `${WORKSPACE}/parameters.yaml`.
```shell
mlcube run --task train --platform docker
```
