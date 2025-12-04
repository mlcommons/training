# MLCube for Flux.1-schnell

MLCube™ GitHub [repository](https://github.com/mlcommons/mlcube). MLCube™ [wiki](https://mlcommons.github.io/mlcube/).

## Project setup

An important requirement is that you must have Docker installed.

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install pip==24.0 && pip install mlcube-docker
# Fetch the implementation from GitHub
git clone https://github.com/mlcommons/training && cd ./training
git fetch origin pull/839/head:feature/mlcube_flux && git checkout feature/mlcube_flux
cd ./text_to_image/mlcube
```

Inside the mlcube directory run the following command to check implemented tasks.

```shell
mlcube describe
```

###  Extra requirements

You need to download the `torchtitan` git submodule:

```shell
git submodule update --init --recursive
```

You also need accept the license for the [FLUX schnell model](https://huggingface.co/black-forest-labs/FLUX.1-schnell) on Hugginface.

Finally, to be able to download all the models you will need to get a token from [Hugginface](https://huggingface.co/settings/tokens).

**Note**: Make sure that when creating the token you select:

* Read access to contents of all public gated repos you can access

After that you can set a new enviroment variable, like this:

```shell
export HUGGING_FACE_HUB_TOKEN="YOUR_TOKEN"
```

### MLCube tasks

* Demo tasks:

Download demo dataset and models.

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

**Note**: To rebuild the image use the flag: `-Pdocker.build_strategy=always` during the `mlcube run` command.
