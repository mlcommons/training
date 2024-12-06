# MLCube for Graph Neural Network

MLCube™ GitHub [repository](https://github.com/mlcommons/mlcube). MLCube™ [wiki](https://mlcommons.github.io/mlcube/).

## Project setup

An important requirement is that you must have Docker installed.

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install pip==24.0 && pip install mlcube-docker
# Fetch the implementation from GitHub
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/762/head:feature/mlcube_graph_nn && git checkout feature/mlcube_graph_nn
cd ./graph_neural_network/mlcube
```

Inside the mlcube directory run the following command to check implemented tasks.

```shell
mlcube describe
```

### MLCube tasks

* Core tasks:

Download dataset.

```shell
mlcube run --task=download_data -Pdocker.build_strategy=always
```

Process dataset.

```shell
mlcube run --task=process_data -Pdocker.build_strategy=always
```

Train GNN.

```shell
mlcube run --task=train -Pdocker.build_strategy=always
```

* Demo tasks:

Download demo dataset.

```shell
mlcube run --task=download_demo -Pdocker.build_strategy=always
```

Run demo training.

```shell
mlcube run --task=demo -Pdocker.build_strategy=always
```

### Execute the complete pipeline

You can execute the complete pipeline with one single command.

* Core pipeline:

```shell
mlcube run --task=download_data,process_data,train -Pdocker.build_strategy=always
```

* Demo pipeline:

```shell
mlcube run --task=download_demo,demo -Pdocker.build_strategy=always
```
