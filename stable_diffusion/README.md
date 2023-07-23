<h1 align="center">Text to image: Stable Diffusion (SD)</h1>

![](imgs/overview.png)

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
  - [Training](#training)
    - [Single node (with docker)](#single-node-with-docker)
    - [Multi-node (with SLURM)](#multi-node-with-slurm)
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
- [Rules](#rules)
- [BibTeX](#bibtex)

# Getting started
This reference implementation is designed for execution on NVIDIA GPUs.
While it has been rigorously tested on A100 cards, it is expected to
perform well on other NVIDIA GPU models as well.

**Please note that all command instructions provided in this README are presumed to be executed from the following working directory: `<repo_path>/stable_diffusion`**

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
    -v /datasets/laion-400m:/datasets/laion-400m \
    -v /datasets/coco2014:/datasets/coco2014 \
    -v /checkpoints:/checkpoints \
    -v /results:/results \
    mlperf/stable_diffusion bash
```

The launch command mounts This command mounts the following directories with
the structure <host_path>:<mount_path>:
* Benchmark code: `${PWD}:/pwd`
* Laion 400m dataset: `/datasets/laion-400m:/datasets/laion-400m`
* Coco-2017 dataset: `/datasets/coco2014:/datasets/coco2014`
* Checkpoints folder: `/checkpoints:/checkpoints`
* Results folder: `/results:/results`

**Please note that all the commands in the following sections are designed to be executed from within a running Docker container.**

## Downloading the dataset
The benchmark employs two datasets:

1. Training: a subset of [laion-400m](https://laion.ai/blog/laion-400-open-dataset)
1. Validation: a subset of [coco-2014 validation](https://cocodataset.org/#download)

### Laion 400m
The benchmark uses a CC-BY licensed subset of the Laion400 dataset.

The LAION datasets comprise lists of URLs for original images, paired with the ALT text linked to those images. As downloading millions of images from the internet is not a deterministic process and to ensure the replicability of the benchmark results, submitters are asked to download the subset from the MLCommons storage. The dataset is provided in two formats:

1. Preprocessed moments (recommended):`scripts/datasets/laion400m-filtered-download-moments.sh --output-dir /datasets/laion-400m/webdataset-moments-filtered`
2. Raw images: `scripts/datasets/laion400m-filtered-download-images.sh --output-dir /datasets/laion-400m/webdataset-filtered`

While the benchmark code is compatible with both formats, we recommend using the preprocessed moments to save on computational resources.

For additional information about Laion 400m, the CC-BY subset, and the scripts used for downloading, filtering, and preprocessing the images, refer to the laion-400m section [here](#the-laion400m-subset).

### COCO-2014
The COCO-2014-validation dataset consists of 40,504 images and 202,654 annotations. However, our benchmark uses only a subset of 30,000 images and annotations chosen at random with a preset seed. It's not necessary to download the entire COCO dataset as our focus is primarily on the labels (prompts) and the inception activation for the corresponding images (used for the FID score).

To ensure reproducibility, we ask the submitters to download the relevant files from the MLCommons storage:

```bash
scripts/datasets/coco2014-validation-download-prompts.sh --output-dir /datasets/coco2014
scripts/datasets/coco2014-validation-download-stats.sh --output-dir /datasets/coco2014
```

While the benchmark code can work with raw images, we recommend using the preprocessed inception weights to save on computational resources.

For additional information about COCO-2014, the creation of weights, and the validation process, refer here.

For additional information about the validation process and the used metrics, refer to the validation section here [here](#validation).

## Downloading the checkpoints

The benchmark utilizes several network architectures for both the training and validation processes:

1. **Stable Diffusion**: This component leverages StabilityAI's 512-base-ema.ckpt checkpoint from HuggingFace. While the checkpoint includes weights for the UNet, VAE, and OpenCLIP text embedder, the UNet weights are not used and are discarded when loading the weights. The checkpoint can be downloaded with the following command:
```bash
scripts/checkpoints/download_sd.sh --output-dir /checkpoints/sd
```
2. **Inception**: The Inception network is employed during validation to compute the Fréchet Inception Distance (FID) score. The necessary weights can be downloaded with the following command:
```bash
scripts/checkpoints/download_inception.sh --output-dir /checkpoints/inception
```
3. **OpenCLIP ViT-H-14 Model**: This model is utilized for the computation of the CLIP score. The required weights can be downloaded using the command:
```bash
scripts/checkpoints/download_clip.sh --output-dir /checkpoints/clip
```

The aforementioned scripts will handle both the download and integrity verification of the checkpoints.

For more information about the various components of the benchmark, kindly refer to [The model](#the-model) section.

## Training

### Single node (with docker)
To initiate a single-node training run, execute the following command from within a running mlperf/stable_diffusion container:

```bash
./run_and_time.sh \
  --num-nodes 1 \
  --gpus-per-node 8 \
  --checkpoint /checkpoints/sd/512-base-ema.ckpt \
  --results-dir /results \
  --config configs/train_512_moments.yaml
```
If you prefer to train using raw images, consider utilizing the `configs/train_512.yaml` configuration file.

### Multi-node (with SLURM)
Given the extended duration it typically takes to train the Stable Diffusion model, it's often beneficial to employ multiple nodes for expedited training. For this purpose, we provide rudimentary Slurm scripts to submit multi-node training batch jobs. Use the following command to submit a batch job:
```bash
scripts/slurm/sbatch.sh \
  --num-nodes 8 \
  --gpus-per-node 8 \
  --checkpoint /checkpoints/sd/512-base-ema.ckpt \
  --config configs/train_512_moments.yaml \
  --results-dir configs/train_512_moments.yaml \
  --container mlperf/stable_diffusion
```

Given the substantial variability among Slurm clusters, users are encouraged to review and adapt these scripts to fit their specific cluster specifications.

In any case, the dataset and checkpoints are expected to be available to all the nodes.

# Benchmark details

## The datasets
### Laion 400m

[Laion-400m](#[the-model](https://laion.ai/blog/laion-400-open-dataset/)) is a rich dataset of 400 million image-text pairs, crafted by the Laion project. The benchmark uses a relatively small subset of this dataset, approximately 6.1M images, all under a CC-BY license.

To establish a fair benchmark and assure reproducibility of results, we request that submitters download either the preprocessed moments or raw images from the MLCommons storage using the scripts provided [here](#laion-400m). These images and moments were generated by following these steps:

1. Download the metadata: `scripts/datasets/laion400m-download-metadata.sh --output-dir /datasets/laion-400m/metadata`
2. Filter the metadata based on LICENSE information: `scripts/datasets/laion400m-filter-metadata.sh --input-metadata-dir /datasets/laion-400m/metadata --output-metadata-dir /datasets/laion-400m/metadata-filtered`
3. Fownload the filtered subset: `scripts/datasets/laion400m-download-dataset --metadata-dir /datasets/laion-400m/metadata-filtered --output-dir /datasets/laion-400m/webdataset-filtered`
4. Preprocess the images to latents `scripts/datasets/laion400m-convert-images-to-moments.sh --input-folder /datasets/laion-400m/webdataset-filtered --output-dir /datasets/laion-400m/webdataset-moments-filtered`

### COCO 2014

The Common Objects in Context ([COCO](https://cocodataset.org/)) dataset is a versatile resource that provides object detection, segmentation, and captioning tasks. This dataset comprises an extensive validation set with over 41,000 images, carrying more than 200,000 labels.

For our benchmark, we adopt a randomly selected subset of 30,000 images and their corresponding labels, using a preset seed of 2023. As our main focus is to calculate the FID (Fréchet Inception Distance) score, it's unnecessary to download the complete dataset. The definitive validation prompts along with the FID statistics can be downloaded following the instructions [here](#coco-2014).

We achieved this subset by following these steps:

1. download coco-2014 validation dataset: `scripts/datasets/coco-2014-validation-download.sh --output-dir /datasets/coco2014`
2. create the validation subset, and resize the images to 512x512: `scripts/datasets/coco-2014-validation-split-resize.sh --input-images-path /datasets/coco2014/val2014 --input-coco-captions /datasets/coco2014/annotations/captions_val2014.json --output-images-path /datasets/coco2014/val2014_512x512_30k --output-tsv-file /datasets/coco2014/val2014_30k.tsv`
3. generate the FID statistics: `scripts/datasets/generate-fid-statistics.sh --dataset-dir /datasets/coco2014/val2014_512x512_30k --output-file /datasets/coco2014/val2014_512x512_30k_stats.npz`

## The Model
Stable Diffusion v2 is a latent diffusion model which combines an autoencoder with a diffusion model that is trained in the latent space of the autoencoder. During training:

* Images are encoded through an encoder, which turns images into latent representations. The autoencoder uses a relative downsampling factor of 8 and maps images of shape 512 x 512 x 3 to latents of shape 64 x 64 x 4
* Text prompts are encoded through the OpenCLIP-ViT/H text-encoder, the output embedding vector has a lengh of 1024.
* The output of the text encoder is fed into the UNet backbone of the latent diffusion model via cross-attention.
* The loss is a reconstruction objective between the noise that was added to the latent and the prediction made by the UNet. We also use the so-called v-objective, see https://arxiv.org/abs/2202.00512.

The UNet backbone in our model serves as the sole trainable component, initialized from random weights. Conversely, the weights of both the image and text encoders are loaded from a pre-existing checkpoint and kept static throughout the training procedure

Although our benchmark aims to adhere to the original Stable Diffusion v2 implementation as closely as possible, it's important to note some key deviations:
1. The group norm of the UNet within our code uses a group size of 16 instead of the 32 used in the original implementation. This adjustment can be found in our code at this [link](https://github.com/ahmadki/training/blob/master/stable_diffusion/ldm/modules/diffusionmodules/util.py#L209)

### UNet
TODO(ahmadki): give an overview
### VAE
TODO(ahmadki): give an overview
### Text encoder
TODO(ahmadki): give an overview
## Validation metrics
### FID
FID is a measure of similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. FID is calculated by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two Gaussians fitted to feature representations of the Inception network. A lower FID implies a better image quality.

Further insights and an independent evaluation of the FID score can be found in [Are GANs Created Equal? A Large-Scale Study.](https://arxiv.org/abs/1711.10337)

### CLIP
CLIP is a reference free metric that can be used to evaluate the correlation between a caption for an image and the actual content of the image, it has been found to be highly correlated with human judgement. A higher CLIP Score implies that the caption matches closer to image.

# Reference runs
The benchmark is expected to have the following convergence profile:
TODO(ahmadki)
# Rules
The benchmark rules can be found [here](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc)


# BibTeX
```
@InProceedings{Rombach_2022_CVPR,
    author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
    title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10684-10695}
}
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
@misc{Seitzer2020FID,
  author={Maximilian Seitzer},
  title={{pytorch-fid: FID Score for PyTorch}},
  month={August},
  year={2020},
  note={Version 0.3.0},
  howpublished={\url{https://github.com/mseitzer/pytorch-fid}},
}
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}
```
