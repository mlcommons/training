# GPT-OSS-20B Pretraining Benchmark

GPT-OSS 20B (Mixture of Experts)

## Overview

This benchmark trains a 20B parameter GPT model with Mixture of Experts (MoE) architecture using the Primus framework on AMD and NVIDIA GPUs.

# 1. Setup Docker Image


Run the following build command from this directory. The build process will take a while to complete.

```bash
# From gpt-oss-20b/primus directory
docker build -t rocm/amd-mlperf:gpt_oss_20b_training_5.1 .
```

# 2. Prepare Dataset

The current codebase uses the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4) for training and evaluation.

## Download Preprocessed Data

The pre-tokenized dataset is available for download. Navigate to your desired download directory and run the following commands:

```bash
# Create desired download directory with the right permission 
cd /data/gpt_oss_20b

# Download training and validation data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
```

After download, you should see files with the following naming conventions:
- Training: `c4-train.en_6_text_document.bin` and `.idx`
- Validation: `c4-validation-91205-samples.en_text_document.bin` and `.idx`

The data directory is approximately **80 GB**.

# 3. Run Training

## Set Environment Variables

Set the directory for data and results. Ensure `$LOGDIR` has write access.

```bash
export DATADIR=/data/gpt_oss_20b/data
export MODELDIR=/data/gpt_oss_20b/model
export LOGDIR=/data/gpt_oss_20b/results
export CONT=rocm/amd-mlperf:gpt_oss_20b_training_5.1

# Create results directory
mkdir -p $LOGDIR
sudo chmod -R 777 $LOGDIR
```

## Set Configuration

Set appropriate configuration and system-specific hyperparameters based on hardware type:

| Config File | System | GPUs |
|-------------|--------|------|
| `config_MI355X_1x8x1.sh` | MI355X | 1 node × 8 GPUs |
| `config_B200_1x8x1.sh` | B200 | 1 node × 8 GPUs |

```bash
source config_MI355X_1x8x1.sh
```

## Launch Training

### Docker
#### Single Run

```bash
export NEXP=1
bash run_with_docker.sh
```

#### Multiple Runs (for submission)

```bash
export NEXP=10
bash run_with_docker.sh
```

### SLURM

```bash
sbatch -A <account> -p <partition> -t <time_limit> run.sub
```

After completion, logs will be available under `$LOGDIR`.

# 4. Quality Metrics

## Target loss

3.34

## Quality Metric

Validation loss (log perplexity)

## Evaluation Frequency

Evaluation every **12,288 samples** (768 iterations with GBS=16)

## Evaluation Thoroughness

We evaluate using the first **1,024 samples** from the validation dataset.

# 5. Model Architecture

| Parameter | Value |
|-----------|-------|
| Model Size | 20B parameters |
| Architecture | GPT with Mixture of Experts |
| Sequence Length | 8192 |
| Expert Parallelism | 8 |

# 6. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Micro Batch Size | 2 |
| Global Batch Size | 16 |
| Learning Rate | 8e-4 |
| LR Schedule | Cosine decay with warmup |
| Weight Decay | 0.1 |
| Adam β1, β2, eps | 0.9, 0.95, 1e-5 |
| Max Training Iterations | 1,200,000 |

# 7. Directory Structure

```
gpt-oss-20b/primus/
├── conf/                       # Configuration files
│   └── gpt_oss_20B-pretrain.yaml
├── src/                        # Training source code
│   └── train.py
├── config_MI355X_1x8x1.sh      # System configuration (MI355 - AMD)
├── config_B200_1x8x1.sh        # System configuration (B200 - NVIDIA)
├── Dockerfile                  # Dockerfile (MI355 - AMD)
├── Dockerfile.nvidia           # Dockerfile (B200 - NVIDIA)
└── requirements.txt            # Python dependencies (includes primus-mllog)
```
# 8. Approximnate runtime

Approximate train time to convergence is ~6.5 hours.
