# MLCube for Stable Diffusion

MLCube™ GitHub [repository](https://github.com/mlcommons/mlcube). MLCube™ [wiki](https://mlcommons.github.io/mlcube/).

## Project setup

An important requirement is that you must have Docker installed.

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker
# Fetch the implementation from GitHub
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/696/head:feature/mlcube_sd && git checkout feature/mlcube_sd
cd ./stable_diffusion/mlcube
```

Inside the mlcube directory run the following command to check implemented tasks.

```shell
mlcube describe
```

### MLCube tasks

* Core tasks:

Download dataset.

```shell
mlcube run --task=download_data
```

Download models.

```shell
mlcube run --task=download_models
```

Train.

```shell
mlcube run --task=train
```

* Demo tasks:

Download demo dataset.

```shell
mlcube run --task=download_demo
```

Download models.

```shell
mlcube run --task=download_models
```

Train demo.

```shell
mlcube run --task=demo
```

### Execute the complete pipeline

You can execute the complete pipeline with one single command.

* Core pipeline:

```shell
mlcube run --task=download_data,download_models,train
```

* Demo pipeline:

```shell
mlcube run --task=download_demo,download_models,demo
```

**Note**: To rebuild the image use the flag: `-Pdocker.build_strategy=always` during the `mlcube run` command.
