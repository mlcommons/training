# Bert benchmark

## MLCube execution

### Project setup
```Python
# Create Python environment and install MLCube Docker runner
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker
```

## Clone Training repo and go to Bert directory
```
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

Also, users can override the workspace directory by using:

```bash
mlcube run --task=download --workspace=absolute_path_to_custom_dir
```

We are targeting pull-type installation, so MLCube images should be available on docker hub. If not, try this:

```bash
mlcube run ... -Pdocker.build_strategy=always
```