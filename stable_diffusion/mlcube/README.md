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

Download dataset.

```shell
mlcube run --task=download_demo
```

Process dataset.

```shell
mlcube run --task=download_models
```

Train SSD.

```shell
mlcube run --task=demo
```

### Execute the complete pipeline

You can execute the complete pipeline with one single command.

```shell
mlcube run --task=download_demo,download_models,demo
```