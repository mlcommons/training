# MLCube for Bert

MLCube™ GitHub [repository](https://github.com/mlcommons/mlcube). MLCube™ [wiki](https://mlcommons.github.io/mlcube/).

## Project setup

An important requirement is that you must have Docker installed.

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker
# Fetch the implementation from GitHub
git clone https://github.com/mlcommons/training && cd ./training/language_model/tensorflow/bert
```

Go to mlcube directory and study what tasks MLCube implements.
```shell
cd ./mlcube
mlcube describe
```

### MLCube tasks

Download dataset.

```shell
mlcube run --task=download_data -Pdocker.build_strategy=always
```

Process dataset.
```shell
mlcube run --task=process_data -Pdocker.build_strategy=always
```

Train SSD.
```shell
mlcube run --task=train -Pdocker.build_strategy=always
```

Run compliance checker.
```shell
mlcube run --task=check_logs -Pdocker.build_strategy=always
```

### Execute the complete pipeline

You can execute the complete pipeline with one single command.
```shell
mlcube run --task=download_data,process_data,train,check_logs -Pdocker.build_strategy=always
```