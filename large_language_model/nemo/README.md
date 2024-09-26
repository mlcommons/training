# 1. Problem 

Large Language Model pretraining - Llama 3.1 405B

# 2. Directions

### Steps to configure machine

To use this repository, please install a supported version of PyTorch with GPU support (python 3.10, pytorch 2.4, cuda 12.5, and nccl 2.22.3 and above) and NVIDIA APEX. **Slurm-based clusters are required to run the reference**. 

We recommend using the latest NeMo FW container. The latest tested compatible version is `nvcr.io/nvidia/nemo:dev`).

#### Container Setup

All of the following codes are assumed to be run within a container. A [Dockerfile](./Dockerfile) is available for building containers on top of `nvcr.io/nvidia/nemo:dev`. 

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

The current codebase is still using GPT3's train/val datasets and SentencePieceModel tokenizer. Please refer to [GPT3 instructions](https://github.com/mlcommons/training/tree/master/large_language_model/megatron-lm#preprocessed-data-download) to download preprocessed datasets and SPM checkpoints. 

### Steps to run and time

To train Llama 3.1 405B, we need to fill out all fields in [config.sh](./config.sh). This file contains all configurations for Slurm cluster access and job submission configurations, directory mappings, containers, and model configurations. 

Once the `config.sh` is properly filled, we run the following code snippet **inside the container**:

```bash
source config.sh
bash run_llama31.sh
```

# 3. Dataset/Environment
### Publication/Attribution

We use the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4). 

### Data preprocessing

To be filled. For now, please refer to [GPT3 data preprocessing instructions](https://github.com/mlcommons/training/tree/master/large_language_model/megatron-lm#dataset-preprocessing). 

### Training and test data separation

To be determined. For now, we are using the default split from the C4 dataset. 

### Training data order

To be determined. 

### Test data order

To be determined. 

# 4. Model
### Publication/Attribution

The model largely follows the Llama 3.1 405B [paper](https://arxiv.org/abs/2407.21783). The main difference is that the model parameters is *to be determined from experiments*. 

### Model details

| Config | Value |
| :-- | :-- | 
| Embedding | RoPE + parameter adjustments |
| # Layers | 126 | 
| Attention Type | GQA |
| # Attn Heads | 128 | 
| Key/Value Heads | 8 | 
| Model Dimension | 16,384 |
| Hidden Dimension | 53248 |
| Activation | SwiGLU | 
| Normalization | RMSNorm |  
| Tokenizer | TokTokenizer |
| Vocab size | 128,000 |  
| Context Length | 8192 |

### Optimizer

Adam

# 5. Quality
### Quality metric

Log Perplexity

### Quality target

To be determined. 

### Evaluation frequency

To be determined. 

### Evaluation thoroughness

To be determined. 