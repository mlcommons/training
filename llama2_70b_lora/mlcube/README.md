# MLCube for Llama 2

MLCube™ GitHub [repository](https://github.com/mlcommons/mlcube). MLCube™ [wiki](https://mlcommons.github.io/mlcube/).

## Project setup

An important requirement is that you must have Docker installed.

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker
# Fetch the implementation from GitHub
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/749/head:feature/mlcube_llama2 && git checkout feature/mlcube_llama2
cd ./llama2_70b_lora/mlcube
```

Inside the mlcube directory run the following command to check implemented tasks.

```shell
mlcube describe
```

### Extra requirements

Install Rclone in your system, by following [these instructions](https://rclone.org/install/).

MLCommons hosts the model for download exclusively by MLCommons Members. You must first agree to the [confidentiality notice](https://docs.google.com/forms/d/e/1FAIpQLSc_8VIvRmXM3I8KQaYnKf7gy27Z63BBoI_I1u02f4lw6rBp3g/viewform).

When finishing the previous form, you will be redirected to a Drive folder containing a file called `CLI Download Instructions`, follow the instructions inside that file up to step: `#3 Authenticate Rclone with Google Drive`.

When finishing this step a configuration file for Rclone will contain the necessary data to download the dataset and models. To check where this file is located run the command:

```bash
 rclone config file
 ```

 **Default:** `~/.config/rclone/rclone.conf`

Finally copy that file inside the `workspace` folder that is located in the same path as this readme, it must have the name `rclone.conf`.

### MLCube tasks

* Core tasks:

Download dataset.

```shell
mlcube run --task=download_data -Pdocker.build_strategy=always
```

Train.

```shell
mlcube run --task=train -Pdocker.build_strategy=always
```

* Demo tasks:

Here is a video explaining the demo steps:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/1Y9q-nltI8U/0.jpg)](https://www.youtube.com/watch?v=1Y9q-nltI8U)

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

* Core pipeline:

```shell
mlcube run --task=download_data,train -Pdocker.build_strategy=always
```

* Demo pipeline:

```shell
mlcube run --task=download_demo,demo -Pdocker.build_strategy=always
```
