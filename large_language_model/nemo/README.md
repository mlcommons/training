# 1. Problem 

Large Language Model pretraining - Llama 3.1 405B

# 2. Directions

### Steps to configure machine

To use this repository, please install a supported version of PyTorch with GPU support (python 3.10, pytorch 2.4, cuda 12.5, and nccl 2.22.3 and above) and NVIDIA APEX. **Slurm-based clusters are required to run the reference**. 

We recommend using the latest NeMo FW container. The latest tested compatible version is `nvcr.io/nvidia/nemo:24.12-rc0`).

#### Container Setup

All of the following codes are assumed to be run within a container. A [Dockerfile](./Dockerfile) is available for building containers on top of `nvcr.io/nvidia/nemo:24.12-rc0`. 

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

The current codebase is still using GPT3's train/val datasets and SentencePieceModel tokenizer. Please refer to [GPT3 instructions](https://github.com/mlcommons/training/tree/master/large_language_model/megatron-lm#preprocessed-data-download) to download **the raw C4 dataset** that we can preprocess later. 

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

We use the Mixtral 8x22B tokenizer from [HuggingFace/MistralAI](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1). 

### Preprocessed data download

The pre-tokenized dataset and the tokenizer are available to download from the S3 bucket. You can download this data from the bucket using RClone as follows: 

To run Rclone on Windows, you can download the executable here. To install Rclone on Linux/macOS/BSD systems, run:

```
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```

Once Rclone is installed, run the following command to authenticate with the bucket:

```
rclone config create mlc-training s3 provider=Cloudflare access_key_id=76ea42eadb867e854061a1806220ee1e secret_access_key=a53625c4d45e3ca8ac0df8a353ea3a41ffc3292aa25259addd8b7dc5a6ce2936 endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
```

You can then navigate in the terminal to your desired download directory and run the following commands to download the dataset and checkpoints: 

#### Dataset

```
# Replace this path with your desired path on the machine
export PREPROCESSED_PATH="./"
rclone copy mlc-training:mlcommons-training-wg-public/llama3_1/datasets/preprocessed_c4 $PREPROCESSED_PATH -P
```

After the download is complete, you should see files with the following naming conventions under `PREPROCESSED_PATH`, ending with both `.idx` and `.bin`: 
- Training partitions: `c4-train.en_<number>_text_document`
- Validation partitions: `c4-validation.en_text_document`

#### Tokenizer

```
# Replace this path with your desired path on the machine
export TOKENIZER_PATH="./"
rclone copy mlc-training:mlcommons-training-wg-public/llama3_1/datasets/tokenizer $TOKENIZER_PATH -P
```

After the download is complete, you should see five files under `TOKENIZER_PATH`: 
- `special_tokens_map.json`
- `tokenizer.json`
- `tokenizer.model`
- `tokenizer.model.v1`
- `tokenizer_config.json`

### Training and test data separation

To be determined. For now, we are using the default split from the C4 dataset. 

### Training data order

To be determined. Current plan is to use the last 256 of 1024 files (shards 6 and 7) for the benchmarked area. 

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
| Tokenizer | TikTokenizer |
| Vocab size | 128,000 |  
| Context Length | 8192 |


### Checkpoint download and conversion

To be determined. For now, we are not using Llama 3.1 default checkpoint. 

~~To experiment with a given checkpoint, we have added a `--ckpt` argument that loads the pretrained checkpoint from a **NeMo checkpoint path**, which requires some checkpoint format conversion if the original checkpoint is in LlamaStack or HuggingFace format.~~

#### Saving and restoring a checkpoint

Large runs might need to span across multiple Slurm jobs, and we need to save and load checkpoints with contexts so that training can resume between jobs. To support this, we have added some environment variables. Please refer to `config.sh` for more details. 

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


# 6. Other

### Data Preprocessing

Here are the instructions to prepare the preprocessed dataset from scratch. Data preprocessing is already done and the final dataset can be accessed by following instructions in the [Preprocessed data download]() section. 

#### Tokenizer

We use Mixtral 8x22B tokenizer in this benchmark. Tokenizer files can be downloaded [here](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/tree/main). Only the five files containing tokenizer-related contents (`special_tokens_map.json`, `tokenizer.json`, `tokenizer.model`, `tokenizer.model.v1`, `tokenizer_config.json`) are needed. 

#### Run Data preprocessing

Run the following commands to merge all 1024 training files into 8 `json.gz` files and all 8 validation files into a single `json.gz` file. Each of the `json.gz` files will be preprocessed into a pair of megatron dataset files (`.bin` and `.idx`). 

```bash
export C4_PATH=""
export MERGED_C4_PATH=""

bash consolidate_data.sh
```

After the data consolidation is done, we can run this [script](./utils/preprocess.sh) to perform preprocessing. To run the preprocessing script, we need to use the following commands: 

```bash
# fill in the built container path here
export CONT_IMAGE_URL=""
# pass in the folder path that contains the Mixtral tokenizer here
# please refer to the tokenizer section above for more details
export TOKENIZER_PATH=""
# pass in the merged file path here
export MERGED_C4_PATH=""
# this path is used for storing the preprocessed .bin and .idx files
export PREPROCESSED_PATH=""

sbatch preprocess.sh
```
