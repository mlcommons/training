<h1 align="center">Text to image: Stable Diffusion (SD)</h1>

Welcome to the reference implementation for the MLPerf text to image
benchmark, utilizing the Stable Diffusion (SD) model.
Our repository prioritizes transparency, reproducibility, reliability,
and user-friendliness. While we encourage the wider audience to explore
other implementations that might provide an extended range of features,
our focus is on delivering a clear and reliable base from which to
understand the fundamentals of the Stable Diffusion model.

- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building the docker image](#building-the-docker-image)
  - [Launching the container](#launching-the-container)
  - [Downloading the dataset](#downloading-the-dataset)
    - [Laion 400m](#laion-400m)
    - [COCO-2014](#coco-2014)
  - [Downloading the checkpoints](#downloading-the-checkpoints)
    - [Stable Diffusion](#stable-diffusion)
    - [Inception (FID score)](#inception-fid-score)
    - [CLIP (CLIP score)](#clip-clip-score)
  - [Training](#training)
    - [Single node (with docker)](#single-node-with-docker)
    - [Multi-node (with SLURM)](#multi-node-with-slurm)
  - [Validation](#validation)
    - [Single node (with docker)](#single-node-with-docker-1)
    - [Multi-node (with SLURM)](#multi-node-with-slurm-1)
- [Benchmark details](#benchmark-details)
  - [The datasets](#the-datasets)
    - [Laion 400m](#laion-400m-1)
    - [COCO 2014](#coco-2014-1)
  - [The Model](#the-model)
    - [UNet](#unet)
    - [VAE](#vae)
    - [Text encoder](#text-encoder)
  - [Validation metrics](#validation-metrics)
    - [FID](#fid)
    - [CLIP](#clip)
- [Reference runs](#reference-runs)
- [Publication/Attribution](#publicationattribution)

# Getting started
This reference implementation is designed for execution on NVIDIA GPUs.
While it has been rigorously tested on A100 cards, it is expected to
perform well on other NVIDIA GPU models as well.

** Please note that all command instructions provided in this README are
presumed to be executed from the following working
directory: `<repo_path>/stable_diffusion` **

## Prerequisites
The recommended way to run the reference is with docker and/or Slurm.
Tset up your system correctly, you'll need to install the
following components:


1. NVIDIA Drivers. Can be installed by 
downloading the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).
2. [Docker](https://docs.docker.com/engine/install/)
3. [NVIDIA Container Runtime](https://github.com/NVIDIA/nvidia-docker)

In addition to the above requirements, it's recommended to use
[Slurm](https://developer.nvidia.com/slurm) for Multi-node training.
Please note, the installation and configuration of Slurm can vary based
on your cluster setup, and is not covered in this README.

## Building the docker image
Once you've satisfied all the prerequisites, you can proceed to build the
benchmark Docker image. This can be accomplished with the
following command:

```bash
docker build -t mlperf/stable_diffusion .
```

## Launching the container
To launch the Docker container, you can use the following command:

```bash
docker run --rm -it --gpus=all --ipc=host \
    --workdir /pwd \
    -v ${PWD}:/pwd \
    -v /datasets/laion2B-en-aesthetic:/datasets/laion2B-en-aesthetic \
    -v /datasets/coco2014:/datasets/coco2014 \
    -v /checkpoints:/checkpoints \
    -v /results:/results \
    mlperf/stable_diffusion bash
```

The launch command mounts This command mounts the following directories with
the structure <host_path>:<mount_path>:
* Benchmark code: `${PWD}:/pwd`
* Laion aesthetic dataset:
`/datasets/laion2B-en-aesthetic:/datasets/laion2B-en-aesthetic`
* Coco-2017 dataset: `/datasets/coco2014:/datasets/coco2014`
* Checkpoints folder: `/checkpoints:/checkpoints`
* Results folder: `/results:/results`

** Please note that all the commands in the following sections are designed to
be executed from within a running Docker container. **

## Downloading the dataset
The benchmark employs two datasets:

1. Training:
a subset of [laion-400m](https://laion.ai/blog/laion-400-open-dataset)
1. Validation:
a subset of [coco-2014 validation](https://cocodataset.org/#download)

### Laion 400m
** TODO(ahmadki): This README presumes that the training dataset is Laion-400m. However, the final dataset choice will be decided upon at the time of the RCP submission. **

The benchmark uses a CC-BY licensed subset of the Laion400 dataset.

The LAION datasets comprise lists of URLs for original images, paired with the ALT text linked to those images. As downloading millions of images from the internet is not a deterministic process and to ensure the replicability of the benchmark results, submitters are asked to download the subset from the MLCommons storage. The dataset is provided in two formats:

** TODO(ahmadki): The scripts will be added once the dataset is uploaded to the MLCommons storage. **

1. Preprocessed latents (recommended):`scripts/datasets/download_laion400m-ccby-latents.sh --output-dir /datasets/laion-400m/ccby_latents_512x512`
2. Raw images: `scripts/datasets/download_laion400m-ccby-images.sh --output-dir /datasets/laion-400m/ccby_images`

While the benchmark code is compatible with both formats, we recommend using the preprocessed latents to save on computational resources.

For additional information about Laion 400m, the CC-BY subset, and the scripts used for downloading, filtering, and preprocessing the images, refer to the laion-400m section [here](#the-laion400m-subset).

### COCO-2014
The COCO-2014-validation dataset consists of 40,504 images and 202,654 annotations. However, our benchmark uses only a subset of 30,000 images and annotations chosen at random with a preset seed. It's not necessary to download the entire COCO dataset as our focus is primarily on the labels (prompts) and the inception activation for the corresponding images (used for the FID score).

To ensure reproducibility, we ask the submitters to download the relevant files from the MLCommons storage:

```bash
scripts/datasets/download_coco-2014.sh --output-dir /datasets/coco2014/val2014-sd
```

While the benchmark code can work with raw images, we recommend using the preprocessed inception weights to save on computational resources.

For additional information about COCO-2014, the creation of weights, and the validation process, refer here.

For additional information about the validation process and the used metrics, refer to the validation section here [here](#validation).

TODO(ahmadki): Everything from this point forward is a WIP

## Downloading the checkpoints

### Stable Diffusion

### Inception (FID score)
### CLIP (CLIP score)


## Training
### Single node (with docker)
```bash
./run_and_time.sh
```

### Multi-node (with SLURM)
TODO(ahmadki)

## Validation
TODO(ahmadki)

### Single node (with docker)
TODO(ahmadki)

### Multi-node (with SLURM)
TODO(ahmadki)

# Benchmark details

## The datasets
TODO(ahmadki): more informatino about the datasets, contributions, size ...
### Laion 400m
To build the laion-400m CC-BY subset:

1. download the metadata: `scripts/datasets/download_laion400m-metada.sh --output-dir /datasets/laion-400m/metadata`
2. filter the metadata based on LICENSE information: `scripts/datasets/filter_laion400m-metada.sh --input-dir /datasets/laion-400m/metadata --output-dir /datasets/laion-400m/metadata-filtered`
3. download the subset: `scripts/datasets/download_laion400m-dataset.sh --metadata-dir /datasets/laion-400m/metadata-filtered --output-dir /datasets/laion-400m/webdataset-filtered`
4. preprocess the images to latents `scripts/datasets/preprocess_laion400m-dataset.sh --metadata-dir /datasets/laion-400m/metadata-filtered --output-dir /datasets/laion-400m/webdataset-filtered`

### COCO 2014
To build the coco-2104 validation files:

1. download the dataset: `scripts/datasets/download_coco-2014-validation.sh --output-dir /datasets/coco2014`
2. create the validation subset, and resize the images to 512x512: `scripts/datasets/process_coco_validation.sh`
3. generate the inception activations: `scripts/datasets/generate_coco_activations.sh`

## The Model
### UNet
### VAE
### Text encoder
## Validation metrics
### FID
### CLIP

# Reference runs

# Publication/Attribution
