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
- [Quality](#quality)
  - [Quality metric](#quality-metric)
  - [Quality target](#quality-target)
  - [Evaluation frequency](#evaluation-frequency)
  - [Evaluation thoroughness](#evaluation-thoroughness)
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
  --config configs/train_01x08x08.yaml
```
If you prefer to train using raw images, consider utilizing the `configs/train_32x08x02_raw_images.yaml` configuration file.

### Multi-node (with SLURM)
Given the extended duration it typically takes to train the Stable Diffusion model, it's often beneficial to employ multiple nodes for expedited training. For this purpose, we provide rudimentary Slurm scripts to submit multi-node training batch jobs. Use the following command to submit a batch job:
```bash
scripts/slurm/sbatch.sh \
  --num-nodes 32 \
  --gpus-per-node 8 \
  --checkpoint /checkpoints/sd/512-base-ema.ckpt \
  --config configs/train_512_moments.yaml \
  --results-dir configs/train_32x08x02.yaml \
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
3. generate the FID statistics: `scripts/datasets/generate-fid-statistics.sh --dataset-dir /datasets/coco2014/val2014_512x512_30k --output-file /datasets/coco2014/val2014_30k_stats.npz`

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

# Quality
## Quality metric
Both FID and CLIP are used to evaulte the model's quality.

## Quality target
FID<=90 and CLIP>=0.15

## Evaluation frequency
Every 512,000 images, or `CEIL(512000 / global_batch_size)` if 512,000 is not divisible by GBS.

Please refer to the benchmark rules [here](https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc) for the exact evaluation rules.

## Evaluation thoroughness
All the prompts in the [coco-2014](#coco-2014) validation subset.

# Reference runs
The benchmark is expected to have the following convergence profile:

Using `configs/train_32x08x02.yaml`:

|               |           | Run 1            |                  | Run 2            |                   | Run 3            |                  | Run 4            |                  | Run 5            |                 | Run 6            |                   | Run 7            |                   | Run 8            |                   | Run 9            |                   | Run 10           |                   | Run 11           |                  | Run 12           |                   | Run 13           |                   | Run 14           |                  |
|---------------|-----------|------------------|------------------|------------------|-------------------|------------------|------------------|------------------|------------------|------------------|-----------------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|------------------|------------------|-------------------|------------------|-------------------|------------------|------------------|
| # Iterations  | # Images  | FID              | CLIP             | FID              | CLIP              | FID              | CLIP             | FID              | CLIP             | FID              | CLIP            | FID              | CLIP              | FID              | CLIP              | FID              | CLIP              | FID              | CLIP              | FID              | CLIP              | FID              | CLIP             | FID              | CLIP              | FID              | CLIP              | FID              | CLIP             |
| 1000          | 512000    | 240.307753187222 | 0.07037353515625 | 294.573046200695 | 0.042327880859375 | 241.683981119201 | 0.0513916015625  | 244.3093813921   | 0.051513671875   | 223.315586246326 | 0.0750732421875 | 226.574970212965 | 0.057525634765625 | 251.340068446753 | 0.052337646484375 | 248.620622416789 | 0.059051513671875 | 239.375032509416 | 0.052093505859375 | 235.714879858368 | 0.060089111328125 | 216.554181677026 | 0.07000732421875 | 254.65308599743  | 0.047882080078125 | 246.05928942625  | 0.050201416015625 | 245.287937367888 | 0.06011962890625 |
| 2000          | 1024000   | 160.470078270219 | 0.095458984375   | 151.759951633367 | 0.11041259765625  | 153.183910033727 | 0.10162353515625 | 179.805348652356 | 0.08294677734375 | 149.675345136843 | 0.11279296875   | 187.266315858632 | 0.0858154296875   | 151.388620255932 | 0.11322021484375  | 146.280853966833 | 0.10528564453125  | 152.773085803242 | 0.1033935546875   | 166.303450203846 | 0.0897216796875   | 157.262003859026 | 0.10089111328125 | 163.671906693434 | 0.1072998046875   | 150.59871834863  | 0.10504150390625  | 168.392019999909 | 0.10430908203125 |
| 3000          | 1536000   | 116.104976258075 | 0.1297607421875  | 104.82406006546  | 0.1458740234375   | 113.28653970832  | 0.13720703125    | 108.181491001522 | 0.1358642578125  | 120.569476481641 | 0.1312255859375 | 121.507471335783 | 0.1385498046875   | 112.246994199521 | 0.134765625       | 118.687092400973 | 0.1241455078125   | 104.768844637884 | 0.1370849609375   | 109.305393022732 | 0.1436767578125   | 108.186119966125 | 0.1424560546875  | 114.106200291533 | 0.130615234375    | 110.985516643455 | 0.1348876953125   | 128.967376625536 | 0.11767578125    |
| 4000          | 2048000   | 98.6194527639151 | 0.1490478515625  | 93.3023418189487 | 0.1614990234375   | 98.470530998531  | 0.147705078125   | 104.506561457867 | 0.1497802734375  | 117.53438644301  | 0.1346435546875 | 93.9551013202581 | 0.156494140625    | 96.7522546886262 | 0.14990234375     | 93.6238405768778 | 0.14892578125     | 99.2660537095259 | 0.14697265625     | 92.4074720353253 | 0.15283203125     | 102.79009009653  | 0.15576171875    | 106.295253230265 | 0.1375732421875   | 106.89872594564  | 0.150390625       | 101.269312343302 | 0.1561279296875  |
| 5000          | 2560000   | 77.9144665349304 | 0.1700439453125  | 84.3912640535151 | 0.1715087890625   | 74.3932278711953 | 0.1763916015625  | 81.8694462285154 | 0.1590576171875  | 90.8364642085304 | 0.1585693359375 | 85.7692721578059 | 0.1630859375      | 84.7254078568678 | 0.1646728515625   | 82.5745049916586 | 0.1634521484375   | 88.3845314314294 | 0.1591796875      | 76.2159562712626 | 0.1741943359375   | 86.4943102965646 | 0.170654296875   | 89.1185837064392 | 0.1600341796875   | 85.7690462394573 | 0.1629638671875   | 83.403400754612  | 0.162841796875   |
| 6000          | 3072000   | 78.277784436582  | 0.175537109375   | 70.7521662926858 | 0.1824951171875   | 69.5446726528529 | 0.1822509765625  | 78.5287888562515 | 0.172119140625   | 73.100411596327  | 0.1802978515625 | 75.7605175664866 | 0.173583984375    | 71.9332734632299 | 0.177734375       | 78.7097765331263 | 0.1685791015625   | 74.2826761370574 | 0.173583984375    | 76.546361057528  | 0.1759033203125   | 77.3434226893306 | 0.1763916015625  | 78.8033732806207 | 0.1737060546875   | 73.3409144967632 | 0.177978515625    | 76.4804577865017 | 0.1829833984375  |
| 7000          | 3584000   | 61.9517434630711 | 0.1998291015625  | 65.8841791315944 | 0.1937255859375   | 59.758534754348  | 0.195068359375   | 68.2849480866907 | 0.1873779296875  | 67.5589398554567 | 0.1978759765625 | 66.3181053517182 | 0.1839599609375   | 63.4926142518813 | 0.190673828125    | 61.9170226262262 | 0.1905517578125   | 69.5521157432934 | 0.184814453125    | 71.102816810754  | 0.1768798828125   | 68.2832394942013 | 0.19091796875    | 65.4174348723552 | 0.1873779296875   | 63.2371374279834 | 0.19287109375     | 66.3072686144305 | 0.1904296875     |
| 8000          | 4096000   | 57.6910232633396 | 0.19580078125    | 63.9510625822693 | 0.192626953125    | 62.0458245490626 | 0.1903076171875  | 63.3404859303251 | 0.1915283203125  | 68.1056529551962 | 0.1993408203125 | 66.3588662355703 | 0.19384765625     | 61.5250891904713 | 0.1998291015625   | 61.9636640921626 | 0.1943359375      | 55.759511792272  | 0.196533203125    | 59.9278142839585 | 0.2008056640625   | 59.5022074715584 | 0.20458984375    | 62.5511721230049 | 0.1959228515625   | 56.9497804269414 | 0.195556640625    | 61.5091127213477 | 0.190185546875   |
| 9000          | 4608000   | 64.6372463854434 | 0.1925048828125  | 63.3690279269714 | 0.1973876953125   | 58.1806763833258 | 0.203369140625   | 62.998614892463  | 0.19189453125    | 62.1201758106838 | 0.2103271484375 | 64.2889779711819 | 0.190673828125    | 63.9357612486334 | 0.1953125         | 54.3882152248573 | 0.2060546875      | 56.0454761991163 | 0.1982421875      | 71.4737059559803 | 0.1845703125      | 60.5208865396143 | 0.1966552734375  | 59.6999421353521 | 0.1986083984375   | 55.3276841950671 | 0.19970703125     | 56.4976270377332 | 0.2032470703125  |
| 10000         | 5120000   | 59.762565549819  | 0.2086181640625  | 55.4643195068765 | 0.21044921875     | 53.6647322693999 | 0.2054443359375  | 59.3463334480937 | 0.1990966796875  | 54.4330239809199 | 0.2135009765625 | 55.0087139920183 | 0.203857421875    | 57.1277597552926 | 0.204833984375    | 49.239847599091  | 0.2100830078125   | 53.05835413967   | 0.2088623046875   | 55.1455458758192 | 0.2081298828125   | 53.4885730532123 | 0.207763671875   | 58.6086353300705 | 0.2054443359375   | 54.4873456830018 | 0.197265625       | 57.2894338303807 | 0.2022705078125  |

Using `configs/train_32x08x04.yaml`:

|               |           | Run 1            |                  | Run 2            |                   | Run 3            |                  | Run 4            |                   | Run 5            |                  | Run 6            |                   | Run 7            |                  | Run 8            |                   | Run 9            |                   | Run 10           |                    | Run 11           |                   | Run 12           |                  | Run 13           |                   |
|---------------|-----------|------------------|------------------|------------------|-------------------|------------------|------------------|------------------|-------------------|------------------|------------------|------------------|-------------------|------------------|------------------|------------------|-------------------|------------------|-------------------|------------------|--------------------|------------------|-------------------|------------------|------------------|------------------|-------------------|
| # Iterations  | # Images  | FID              | CLIP             | FID              | CLIP              | FID              | CLIP             | FID              | CLIP              | FID              | CLIP             | FID              | CLIP              | FID              | CLIP             | FID              | CLIP              | FID              | CLIP              | FID              | CLIP               | FID              | CLIP              | FID              | CLIP             | FID              | CLIP              |
| 500           | 512000    | 340.118601986399 | 0.0345458984375  | 345.832024994334 | 0.04046630859375  | 338.866903071458 | 0.05609130859375 | 426.617864143804 | 0.034149169921875 | 334.872061080559 | 0.0396728515625  | 439.960494194387 | 0.041351318359375 | 376.763327218143 | 0.0313720703125  | 342.790895799001 | 0.036529541015625 | 547.18671253061  | 0.039031982421875 | 351.895078815387 | 0.0277252197265625 | 343.04031513556  | 0.035980224609375 | 328.479876923655 | 0.0419921875     | 321.099746318961 | 0.0372314453125   |
| 1000          | 1024000   | 223.961567606484 | 0.07330322265625 | 232.917300402437 | 0.057159423828125 | 226.63936355015  | 0.06903076171875 | 230.062675246233 | 0.06732177734375  | 193.949092294986 | 0.084716796875   | 226.70889007593  | 0.07720947265625  | 214.400308061983 | 0.081298828125   | 190.005066809593 | 0.12176513671875  | 205.746223579365 | 0.0953369140625   | 224.200208510141 | 0.068115234375     | 211.447498723363 | 0.0802001953125   | 238.545257722353 | 0.057373046875   | 235.478069634173 | 0.056060791015625 |
| 1500          | 1536000   | 151.10737628716  | 0.11016845703125 | 150.101113881509 | 0.11566162109375  | 134.800112693793 | 0.12103271484375 | 158.111621138263 | 0.1094970703125   | 166.836301701497 | 0.10260009765625 | 149.410719271839 | 0.12164306640625  | 158.375092356384 | 0.10687255859375 | 147.525295062812 | 0.11199951171875  | 127.031851421568 | 0.114990234375    | 158.962418201722 | 0.11505126953125   | 144.255068287832 | 0.113525390625    | 164.992543548194 | 0.10479736328125 | 141.811754285011 | 0.123779296875    |
| 2000          | 2048000   | 124.49736091336  | 0.12841796875    | 117.097835152435 | 0.1282958984375   | 108.370974667191 | 0.152587890625   | 133.81907104026  | 0.1280517578125   | 121.229882811864 | 0.1253662109375  | 110.412332150214 | 0.1422119140625   | 100.932808149808 | 0.1419677734375  | 119.578608307018 | 0.133544921875    | 118.33252832813  | 0.133544921875    | 112.354203644191 | 0.134765625        | 117.135931831381 | 0.14013671875     | 124.541325095207 | 0.1229248046875  | 96.3941054845615 | 0.14990234375     |
| 2500          | 2560000   | 96.9733204361302 | 0.1611328125     | 98.295106421797  | 0.1546630859375   | 89.1922536875995 | 0.1680908203125  | 96.6641498310134 | 0.1571044921875   | 93.8952999670028 | 0.1611328125     | 102.492835331058 | 0.1624755859375   | 89.9220320598469 | 0.1568603515625  | 90.985351002816  | 0.157958984375    | 88.848829608497  | 0.1697998046875   | 94.0665058388316 | 0.15673828125      | 87.9920662633317 | 0.1614990234375   | 96.7143357056873 | 0.148193359375   | 88.7038772396644 | 0.1605224609375   |
| 3000          | 3072000   | 93.3597904033127 | 0.1673583984375  | 88.0347940287123 | 0.1611328125      | 78.7300792641628 | 0.17431640625    | 73.0465712965473 | 0.1798095703125   | 86.9431468184751 | 0.1639404296875  | 87.7669614240029 | 0.1722412109375   | 88.3675919682403 | 0.171630859375   | 82.6417540012699 | 0.1787109375      | 86.7623871245797 | 0.17138671875     | 79.4699853861112 | 0.170654296875     | 72.7502962971125 | 0.177001953125    | 83.9019958654259 | 0.169677734375   | 85.3945042513396 | 0.1646728515625   |
| 3500          | 3584000   | 85.1014052323429 | 0.1712646484375  | 79.6465945046047 | 0.17724609375     | 73.179691682848  | 0.1766357421875  | 72.2186363144172 | 0.17529296875     | 82.2117505009101 | 0.16357421875    | 77.0382476200993 | 0.1776123046875   | 76.2638780498772 | 0.1785888671875  | 82.374727338282  | 0.174560546875    | 73.9219581307538 | 0.181640625       | 75.2863090474383 | 0.1729736328125    | 76.0410092939919 | 0.1761474609375   | 73.0823340750989 | 0.1776123046875  | 68.0820682944884 | 0.1822509765625   |
| 4000          | 4096000   | 67.0934321304885 | 0.18408203125    | 73.6632478577287 | 0.179931640625    | 65.7745415726648 | 0.1893310546875  | 57.5382044168439 | 0.202392578125    | 70.3029895611878 | 0.178955078125   | 63.5395270557031 | 0.1856689453125   | 65.7229965733364 | 0.189697265625   | 63.5817674444506 | 0.194580078125    | 68.9505671827645 | 0.19384765625     | 65.7320230343007 | 0.187255859375     | 60.5971122060028 | 0.2027587890625   | 64.2448592538921 | 0.1851806640625  | 69.7957092926321 | 0.194580078125    |
| 4500          | 4608000   | 67.7145070185802 | 0.1905517578125  | 61.2478434208809 | 0.1939697265625   | 60.257862121379  | 0.1939697265625  | 56.0916643007543 | 0.195556640625    | 63.112236374844  | 0.1904296875     | 61.7597270678272 | 0.1923828125      | 68.7368374504532 | 0.185791015625   | 61.8152633671387 | 0.197265625       | 60.5004491149336 | 0.2005615234375   | 62.5574345222566 | 0.1959228515625    | 62.923525204544  | 0.1995849609375   | 62.0852679830577 | 0.194580078125   | 61.2746769203445 | 0.199951171875    |
| 5000          | 5120000   | 62.8701763107206 | 0.194580078125   | 58.8702801515778 | 0.197265625       | 55.8112946371526 | 0.201904296875   | 58.6735053693167 | 0.1932373046875   | 65.8271859495193 | 0.1854248046875  | 56.2070200654653 | 0.1990966796875   | 60.6945204351451 | 0.1988525390625  | 56.4405121502396 | 0.2056884765625   | 61.6967046304992 | 0.19482421875     | 59.4338652228432 | 0.1962890625       | 55.3596165721984 | 0.2091064453125   | 67.6589663911387 | 0.1868896484375  | 53.2199604152237 | 0.2021484375      |
| 5500          | 5632000   | 61.4585472059858 | 0.1983642578125  | 58.2543886841241 | 0.1995849609375   | 50.8558883425227 | 0.2071533203125  | 51.5331037617053 | 0.2041015625      | 57.3083074689686 | 0.19482421875    | 52.3925368121613 | 0.2099609375      | 57.2835125038846 | 0.200439453125   | 53.6868530762218 | 0.208984375       | 52.2227233111886 | 0.206787109375    | 61.3110736465047 | 0.1966552734375    | 51.6719495239864 | 0.2080078125      | 65.9607437239914 | 0.1888427734375  | 54.6698202033624 | 0.206787109375    |

Using `configs/train_32x08x08.yaml`:

|               |          | Run 1            |                   | Run 2            |                   | Run 3            |                   | Run 4            |                   | Run 5            |                   | Run 6            |                   | Run 7            |                   | Run 8            |                   | Run 9            |                   | Run 10           |                   | Run 11           |                  | Run 12           |                   | Run 13           |                    |
|---------------|----------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|-------------------|------------------|------------------|------------------|-------------------|------------------|--------------------|
| #  Iterations | # Images | FID              | CLIP              | FID              | CLIP              | FID              | CLIP              | FID              | CLIP              | FID              | CLIP              | FID              | CLIP              | FID              | CLIP              | FID              | CLIP              | FID              | CLIP              | FID              | CLIP              | FID              | CLIP             | FID              | CLIP              | FID              | CLIP               |
| 250           | 512000   | 391.39655901198  | 0.048370361328125 | 383.955095156789 | 0.045623779296875 | 375.572664708181 | 0.040283203125    | 372.072899672844 | 0.043426513671875 | 372.901651361397 | 0.034698486328125 | 353.118691841468 | 0.04925537109375  | 342.393139376256 | 0.054412841796875 | 311.78591733597  | 0.058197021484375 | 354.174052398155 | 0.051666259765625 | 374.390968238394 | 0.044097900390625 | 458.052244334111 | 0.03900146484375 | 513.581603241846 | 0.041412353515625 | 428.950254618589 | 0.032989501953125  |
| 500           | 1024000  | 279.186983006922 | 0.045806884765625 | 251.32372064638  | 0.0732421875      | 380.401122648586 | 0.029754638671875 | 265.254012443236 | 0.044219970703125 | 305.165371161357 | 0.043701171875    | 278.005709943431 | 0.041412353515625 | 326.05919184743  | 0.03851318359375  | 285.249900410478 | 0.051910400390625 | 268.047528775676 | 0.06549072265625  | 310.318767239424 | 0.03607177734375  | 268.197820895841 | 0.05615234375    | 276.718579547372 | 0.05157470703125  | 307.082416305895 | 0.0285797119140625 |
| 750           | 1536000  | 233.095664289828 | 0.08306884765625  | 255.574797209144 | 0.061859130859375 | 206.772130562528 | 0.0755615234375   | 221.581035890856 | 0.08935546875     | 245.255653191484 | 0.06085205078125  | 234.983598562262 | 0.058563232421875 | 250.570216717967 | 0.060821533203125 | 259.234382045129 | 0.07977294921875  | 249.22752060747  | 0.05804443359375  | 239.86518048089  | 0.06610107421875  | 219.41761236929  | 0.07098388671875 | 293.547429078577 | 0.05621337890625  | 240.056633721395 | 0.061737060546875  |
| 1000          | 2048000  | 183.214691596374 | 0.09649658203125  | 175.121451173839 | 0.0919189453125   | 201.125958260278 | 0.07904052734375  | 194.480361994534 | 0.10003662109375  | 187.902157253089 | 0.095947265625    | 190.636908095649 | 0.093505859375    | 173.836128053516 | 0.11016845703125  | 198.839618312897 | 0.09100341796875  | 191.57493488944  | 0.09283447265625  | 204.476347827797 | 0.08734130859375  | 190.85390278202  | 0.11077880859375 | 177.090880284724 | 0.099853515625    | 172.779613417181 | 0.0966796875       |
| 1250          | 2560000  | 162.777532690807 | 0.1082763671875   | 147.760423274479 | 0.1129150390625   | 140.85537974249  | 0.12017822265625  | 137.166923984101 | 0.132568359375    | 151.973048193423 | 0.10992431640625  | 148.929026349028 | 0.12249755859375  | 155.895933010689 | 0.12353515625     | 129.288711675567 | 0.137451171875    | 132.886776251458 | 0.125             | 145.39008142017  | 0.12457275390625  | 137.668940540157 | 0.1220703125     | 156.968006561967 | 0.11407470703125  | 142.581989214077 | 0.1314697265625    |
| 1500          | 3072000  | 118.629682612961 | 0.1375732421875   | 106.535941444941 | 0.14697265625     | 114.290550919822 | 0.138916015625    | 108.378854902796 | 0.149169921875    | 98.9089744692479 | 0.1510009765625   | 103.611036790713 | 0.1544189453125   | 131.286037142359 | 0.12109375        | 103.39877184704  | 0.144775390625    | 117.09422144333  | 0.1375732421875   | 105.877016093857 | 0.1439208984375   | 118.102154582684 | 0.1405029296875  | 103.335599019344 | 0.157470703125    | 113.055096586663 | 0.1434326171875    |
| 1750          | 3584000  | 89.4331758025722 | 0.162841796875    | 109.546171022477 | 0.1474609375      | 93.1588864410366 | 0.1578369140625   | 83.5085170500488 | 0.1685791015625   | 94.4197067704322 | 0.1588134765625   | 92.6207403376717 | 0.162841796875    | 114.245383656395 | 0.1517333984375   | 87.25296374262   | 0.1685791015625   | 100.343798309662 | 0.1514892578125   | 87.2525657582866 | 0.1697998046875   | 95.2678809672573 | 0.15966796875    | 101.659175216742 | 0.161376953125    | 94.0385862436482 | 0.15966796875      |
| 2000          | 4096000  | 84.1395342891986 | 0.1728515625      | 86.4941692827467 | 0.171875          | 74.1707863081292 | 0.173583984375    | 87.1637452390254 | 0.1680908203125   | 78.2313975218325 | 0.17431640625     | 83.3830756736107 | 0.172607421875    | 109.383279454652 | 0.1553955078125   | 82.4171094681747 | 0.174560546875    | 87.7106252823979 | 0.1656494140625   | 82.7069497909496 | 0.1707763671875   | 76.3407924798623 | 0.183837890625   | 83.0637053735624 | 0.175048828125    | 94.0479392718985 | 0.1612548828125    |
| 2250          | 4608000  | 70.582001057074  | 0.18505859375     | 74.8823190242059 | 0.181396484375    | 71.1209497479827 | 0.1898193359375   | 70.0543828502346 | 0.184326171875    | 71.8003769855694 | 0.185302734375    | 64.2308369345424 | 0.1934814453125   | 76.4148499308404 | 0.174560546875    | 73.2825963775967 | 0.1800537109375   | 74.3825432407552 | 0.17919921875     | 75.9861941189723 | 0.1817626953125   | 74.1085235586791 | 0.179443359375   | 74.5693447018001 | 0.175537109375    | 77.284322467554  | 0.17822265625      |
| 2500          | 5120000  | 65.3838288644697 | 0.1983642578125   | 87.5473119370954 | 0.1688232421875   | 67.7603158381713 | 0.1884765625      | 62.3535779250927 | 0.1925048828125   | 60.8354202339956 | 0.2000732421875   | 78.4751488606406 | 0.1842041015625   | 68.3657367892087 | 0.185791015625    | 65.0020280359325 | 0.1942138671875   | 67.1935556265445 | 0.1871337890625   | 62.4738416640282 | 0.198974609375    | 64.389505383101  | 0.1939697265625  | 77.7334283478027 | 0.1776123046875   | 68.7218599839536 | 0.1864013671875    |
| 2750          | 5632000  | 59.4060543096938 | 0.1959228515625   | 74.741528787536  | 0.182861328125    | 60.1765143149466 | 0.20068359375     | 58.8862731872101 | 0.197021484375    | 60.4208051048209 | 0.19775390625     | 56.8841221646688 | 0.1993408203125   | 70.8894837982713 | 0.1861572265625   | 71.2384122739707 | 0.1981201171875   | 65.8023543269634 | 0.1947021484375   | 59.4144150742882 | 0.1959228515625   | 58.3607462687932 | 0.204833984375   | 59.8219103754054 | 0.1953125         | 60.7754410616672 | 0.2003173828125    |
| 3000          | 6144000  | 56.6601142266372 | 0.2022705078125   | 68.2707963466305 | 0.1937255859375   | 53.0671317239887 | 0.2049560546875   | 52.5991947034437 | 0.20751953125     | 56.1652745581564 | 0.2054443359375   | 61.7744352999075 | 0.20458984375     | 56.3335107682475 | 0.20654296875     | 64.8515262336457 | 0.193359375       | 67.671189122359  | 0.1947021484375   | 60.6994318650713 | 0.19970703125     | 63.2553342895905 | 0.198974609375   | 63.5162905880337 | 0.195068359375    | 56.9187600001671 | 0.20361328125      |



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
