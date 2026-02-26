# 1. Problem

Large Language Model pretraining - DeepSeek V3 671B

# 2. Directions

### Steps to configure machine

To use this repository, please install a supported version of PyTorch with GPU support and NVIDIA APEX. **Slurm-based clusters are required to run the reference**.

We recommend using the latest PyTorch container. The latest tested compatible version is `nvcr.io/nvidia/pytorch:25.12-py3`.

#### Container setup

All of the following codes are assumed to be run within a container. A [Dockerfile](./Dockerfile) is available for building containers on top of `nvcr.io/nvidia/pytorch:25.12-py3`.

To build the container:

```bash
docker build -t <tag> -f Dockerfile .
```

To launch the container:

```bash
docker run -it --rm \
--network=host --ipc=host \
-v ~/.ssh:/root/.ssh \
<tag> bash
```

Note: it's recommended to map your `.ssh` folder to inside the container, so that it's easier for the code to set up remote cluster access.

### Steps to download and verify data

The current codebase is using C4 dataset for train and evaluation. Please refer to [Section 3](#preprocessed-data-download) for downloading the preprocessed dataset and [Section 6](#data-preprocessing) if you would like to perform manual tokenization.

### Steps to run and time

To train DeepSeek V3 671B, we need to fill out all fields in one of the config files (e.g. [config_GB300_64x4x256xtp1pp4cp1.sh](./config_GB300_64x4x256xtp1pp4cp1.sh)). This file contains all configurations for Slurm cluster access and job submission, directory mappings, containers, and model configurations.

Once the config is properly filled, we run the following code snippet **inside the container**:

```bash
source config_GB300_64x4x256xtp1pp4cp1.sh
bash run_deepseek_v3_671b.sh
```

# 3. Dataset/Environment

### Publication/Attribution

We use the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4).

We use the DeepSeek V3 tokenizer from [HuggingFace/DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V3).

### Preprocessed data download

The pre-tokenized dataset and the tokenizer are available to download from the S3 bucket. You can download this data from the bucket using the [MLCommons R2 Downloader](https://github.com/mlcommons/r2-downloader).

Navigate in the terminal to your desired download directory and run the following commands to download the dataset and checkpoints. More information about the MLCommons R2 Downloader, including how to run it on Windows, can be found [here](https://training.mlcommons-storage.org).

#### Dataset

```bash
# Replace this path with your desired path on the machine
export PREPROCESSED_PATH="./"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d $PREPROCESSED_PATH https://training.mlcommons-storage.org/metadata/deepseek-v3-preprocessed-c4-dataset.uri
```

After the download is complete, you should see files with the following naming conventions under `PREPROCESSED_PATH`, ending with both `.idx` and `.bin`:
- Training partitions: `c4-train.en_<number>_text_document`
- Validation partitions: `c4-validation-91205-samples.en_text_document`

#### Tokenizer

```bash
# Replace this path with your desired path on the machine
export TOKENIZER_PATH="./"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d $TOKENIZER_PATH https://training.mlcommons-storage.org/metadata/deepseek-v3-tokenizer.uri
```

### Training and test data separation

We use the default split from the C4 dataset. This means that we use `c4-train.en_<x>-of-01024.json.gz` files for training, and we use our customized `c4-validation-91205-samples.en.json.gz` for evaluation.

### Training data order

We randomly shuffle the training shards for the benchmarking area.

### Test data order

We use the first sequences in the validation dataset for validation. We **do not shuffle** the validation dataset.

# 4. Model

### Publication/Attribution

The model follows the DeepSeek V3 671B [paper](https://arxiv.org/abs/2412.19437).

### Model details

| Config | Value |
| :-- | :-- |
| # Total Parameters | 671B |
| # Active Parameters | 37B |
| # Layers | 61 |
| Attention Type | MLA (Multi-head Latent Attention) |
| # Attention Heads | 128 |
| # KV Heads | 128 |
| Model Dimension | 7,168 |
| # Routed Experts | 256 |
| # Active Experts | 8 |
| # Shared Experts | 1 |
| Activation | SwiGLU |
| Normalization | RMSNorm |
| Tokenizer | DeepSeek V3 tokenizer |
| Vocab size | 129,280 |
| Context Length | 4,096 |

### Checkpoint download

MLCommons hosts the checkpoint for download **exclusively by MLCommons Members**. You must first agree to the confidentiality notice using your organizational email address, then you will receive a link to download instructions with [MLCommons R2 Downloader](https://github.com/mlcommons/r2_downloader) commands.

#### Saving and restoring a checkpoint

Large runs might need to span across multiple Slurm jobs, and we need to save and load checkpoints with contexts so that training can resume between jobs. To support this, we have added some environment variables. Please refer to the config files for more details.

### Optimizer spec

1. Optimizer type: **AdamW**
2. Warmup steps: 4
3. LR Scheduler's maximum number of steps: 12,000
4. Peak learning rate: 2.4e-5 (for GBS=16384)
5. Minimum learning rate: 1e-8

# 5. Quality

### Quality metric

Validation loss (LM loss)

### Quality target

Validation loss = 2.55

### Evaluation frequency

We perform evaluation every step.

### Evaluation thoroughness

We evaluate using 1,024 samples from our customized validation dataset.

# 6. Other

### Data preprocessing

Here are the instructions to prepare the preprocessed dataset from scratch. Data preprocessing is already done and the final dataset can be accessed by following instructions in the [Preprocessed data download](#preprocessed-data-download) section.

#### Raw data downloading

We use [AllenAI C4](https://huggingface.co/datasets/allenai/c4) dataset for this benchmark. The original zipped **`json.gz`** files can be downloaded by following AllenAI C4's instructions.

#### Run data preprocessing

After downloading the raw data and tokenizer, run the preprocessing script to generate `.bin` and `.idx` files compatible with Megatron-Core's data loader.

### HuggingFace Checkpoint Preprocessing

#### HuggingFace checkpoint downloading

We use the HuggingFace DeepSeek V3 671B checkpoint as the initial checkpoint in this benchmark. The original HuggingFace checkpoint can be downloaded [here](https://huggingface.co/deepseek-ai/DeepSeek-V3).

#### Run model conversion

Assuming that we have downloaded the HuggingFace checkpoint to a `<SRC_PATH>` directory, the checkpoint must be converted to Megatron-Bridge format before training. After conversion is done, set `MODEL_CKPT=<DST_PATH>` when launching the job.
