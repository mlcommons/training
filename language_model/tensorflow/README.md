# Bert benchmark

## MLCube execution

### Project setup

```Python
# Create Python environment 
virtualenv -p python3 ./env && source ./env/bin/activate

# Install MLCube and MLCube docker runner from GitHub repository (normally, users will just run `pip install mlcube mlcube_docker`)
git clone https://github.com/mlcommons/training.git && cd ./training
git fetch origin pull/503/head:feature/bert_mlcube && git checkout feature/bert_mlcube
cd ./language_model/tensorflow
```

## Clone Training repo and go to Bert directory

```bash
git clone https://github.com/mlcommons/training.git && cd ./training
git fetch origin pull/503/head:feature/bert_mlcube && git checkout feature/bert_mlcube
cd ./language_model/tensorflow
```

## Run Bert MLCube on a local machine with Docker runner

```bash
# Run Bert tasks: download, extract, preprocess, generate_tfrecords and train
mlcube run --task download
mlcube run --task extract
mlcube run --task preprocess
mlcube run --task generate_tfrecords
mlcube run --task train
```

We are targeting pull-type installation, so MLCubes should be available on docker hub. If not, try this:

```bash
mlcube run ... -Pdocker.build_strategy=auto
```

Parameters defined in mlcube.yaml can be overridden using: param=input, example:

```bash
mlcube run --task=download data_dir=absolute_path_to_custom_dir
```

Also, users can override the workspace directory by using:

```bash
mlcube run --task=download --workspace=absolute_path_to_custom_dir
```
