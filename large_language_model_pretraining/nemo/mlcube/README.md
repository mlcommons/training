# MLCube for Llama 3.1

MLCube™ GitHub [repository](https://github.com/mlcommons/mlcube). MLCube™ [wiki](https://mlcommons.github.io/mlcube/).

## Project setup

An important requirement is that you must have Docker installed.

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker
# Fetch the implementation from GitHub
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/xxx/head:feature/mlcube_llama3.1 && git checkout feature/mlcube_llama3.1
cd ./large_language_model_pretraining/nemo/mlcube
```

Inside the mlcube directory run the following command to check implemented tasks.

```shell
mlcube describe
```

### Extra requirements

Install Rclone in your system, by following [these instructions](https://rclone.org/install/).

MLCommons hosts the model for download exclusively by MLCommons Members. You must first agree to the [confidentiality notice](https://sites.google.com/view/mlcommons-llama3-1). If you cannot access the form, follow these [intructions](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining/nemo#checkpoint-download).

When finishing the previous form, you will receive an email with access to the Drive folder containing a file called `Llama 3.1 CLI Download Instructions`, follow the instructions inside that file up to step: `3. Authenticate Rclone with Google Drive`.

When finishing this step a configuration file for Rclone will contain the necessary data to download the dataset and models. To check where this file is located run the command:

```bash
rclone config file
```

 **Default:** `~/.config/rclone/rclone.conf`

Finally copy that file inside the `workspace` folder that is located in the same path as this readme, it must have the name `rclone.conf`.

### MLCube tasks

* Demo tasks:

Download demo dataset.

```shell
mlcube run --task=download_demo -Pdocker.build_strategy=always
```

Train demo.

```shell
mlcube run --task=demo -Pdocker.build_strategy=always
```

### Execute the complete pipeline

You can execute the complete pipeline with one single command.

* Demo pipeline:

```shell
mlcube run --task=download_demo,demo -Pdocker.build_strategy=always
```
