# 1. Problem 

Large Language Model pretraining - Llama 3.1 405B

# 2. Directions

### Steps to configure machine

To use this repository, please install a supported version of PyTorch with GPU support (python 3.10, pytorch 2.4, cuda 12.5, and nccl 2.22.3 and above) and NVIDIA APEX. **Slurm-based clusters are required to run the reference**. 

We recommend using the latest NeMo FW container. The latest tested compatible version is `nvcr.io/nvidia/nemo:24.12-rc0`).

#### Container setup

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

The current codebase is using C4 dataset for train and evaluation. Please refer to [Section 3](#preprocessed-data-download) for downloading the preprocessed dataset and [Section 6](#data-preprocessing) if you would like to perform manual tokenization. 

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
rclone copy mlc-training:mlcommons-training-wg-public/common/datasets/c4/mixtral_8x22b_preprocessed $PREPROCESSED_PATH -P
```

After the download is complete, you should see files with the following naming conventions under `PREPROCESSED_PATH`, ending with both `.idx` and `.bin`: 
- Training partitions: `c4-train.en_<number>_text_document`
- Validation partitions: `c4-validation-91205-samples.en_text_document`

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

We use the default split from the C4 dataset. This means that we use `c4-train.<x>-of-01024.json.gz` files (where `768 <= x <= 1023`) for training, and we use our customized `c4-validation-91205-samples.en.json.gz`, which contains the first 91205 samples from the unshuffled C4 validation dataset, for evaluation. 

Notice here that we are using the first 5760 sequences (47,185,920 tokens) from the validation dataset to perform the validation. According to our experiments, the first 91205 samples from the unshuffled C4 dataset yields 47,186,855 tokens, which is the smallest amount of samples needed to yield 47,185,920 tokens. Thus, we have chosen the first 91205 samples as our validation dataset. 

### Training data order

We randomly shuffle the **last 256 of 1024 shards** for the benchmarking area.

### Test data order

We use the first 5,760 sequences (91,205 untokenized samples) in the validation dataset for validation. We **do not shuffle** the validation dataset. 

# 4. Model
### Publication/Attribution

The model largely follows the Llama 3.1 405B [paper](https://arxiv.org/abs/2407.21783). The only difference is: 
- We replace the paper's TikTokenizer with the **Mixtral 8x22b tokenizer** in this benchmark. Please refer to the [Tokenizer](#tokenizer) section for more details.  

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
| Tokenizer | Mixtral 8x22B tokenizer |
| Vocab size | 32,000 |  
| Context Length | 8192 |


### Checkpoint download

MLCommons hosts the checkpoint for download **exclusively by MLCommons Members**. You must first agree to the [confidentiality notice](https://llama3-1.mlcommons.org) using your organizational email address, then you will receive a link to a directory containing Rclone download instructions. _If you cannot access the form but you are part of a MLCommons Member organization, submit the [MLCommons subscription form](https://mlcommons.org/community/subscribe/) with your organizational email address and [associate a Google account](https://accounts.google.com/SignUpWithoutGmail) with your organizational email address. You should then be able to access the confidentiality form using that Google account._

#### Saving and restoring a checkpoint

Large runs might need to span across multiple Slurm jobs, and we need to save and load checkpoints with contexts so that training can resume between jobs. To support this, we have added some environment variables. Please refer to `config.sh` for more details. 

### Optimizer spec

1. Optimizer type: **AdamW**
2. Warmup steps computed as $8000 \times \lceil {1152 \over GBS} \rceil$.
3. LR Scheduler's maximum number of steps computed as $1,200,000 \times \lceil {1152 \over GBS} \rceil$

# 5. Quality
### Quality metric

Log Perplexity

### Quality target

Validation log perplexity = 5.6

### Evaluation frequency

We perform evaluation every **46,080** sequences. 

### Evaluation thoroughness

We evaluate using **5,760** sequences from our customized validation dataset. 


# 6. Other

### Data preprocessing

Here are the instructions to prepare the preprocessed dataset from scratch. Data preprocessing is already done and the final dataset can be accessed by following instructions in the [Preprocessed data download](#preprocessed-data-download) section. 

#### Raw data downloading

We use [AllenAI C4](https://huggingface.co/datasets/allenai/c4) dataset for this benchmark. The original zipped **`json.gz`** files can be downloaded by following AllenAI C4's instruction, and you can download our zipped customized validation dataset from the MLCommons S3 bucket by running the following command: 

```bash
export ORIGINAL_C4_PATH=""

# download the customized zipped validation dataset
rclone copy mlc-training:mlcommons-training-wg-public/common/datasets/c4/original/c4-validation-91205-samples.en.json.gz $ORIGINAL_C4_PATH -P
```

Alternatively, we have also hosted the **unzipped C4 `json`** files on MLCommons S3 bucket. You can download them using the following commands: 

```bash
export ORIGINAL_C4_PATH=""

# download the full C4 files, including all raw train and validations
rclone copy mlc-training:mlcommons-training-wg-public/common/datasets/c4/original/en_json/3.0.1 $ORIGINAL_C4_PATH -P
```

Note that for unzipped JSON files, it is recommended to zip them into `.gz` format before running the data preprocessing. 

#### Prepare tokenizer

We use Mixtral 8x22B tokenizer in this benchmark. Tokenizer files can be downloaded [here](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/tree/main). Only the five files containing tokenizer-related contents (`special_tokens_map.json`, `tokenizer.json`, `tokenizer.model`, `tokenizer.model.v1`, `tokenizer_config.json`) are needed. 

#### Run data preprocessing

Run the following commands to merge all 1024 training files into 8 `json.gz` files, all 8 validation files into a single `json.gz` file, as well as generate our customized validation dataset. Each of the `json.gz` files will subsequently be preprocessed into a pair of megatron dataset files (`.bin` and `.idx`) by our preprocess.sh script. 

```bash
export C4_PATH=""
export MERGED_C4_PATH=""
# more information about this knob can be found in consolidate_data.sh
export N_VALIDATION_SAMPLES=91205

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

# Extra Slurm-related arguments can be provided here
sbatch preprocess.sh
```

### HuggingFace Checkpoint Preprocessing

Here are the instructions to prepare the NeMo-formatted checkpoint from scratch. Checkpoint conversion is already done and the converted checkpoint can be accessed by following instructions in the [Checkpoint download](#checkpoint-download) section. 

#### HuggingFace checkpoint downloading

We use the HuggingFace Llama 3.1 405B checkpoint as the initial checkpoint in this benchmark. Original HuggingFace checkpoint can be downloaded [here](https://huggingface.co/meta-llama/Llama-3.1-405B). **Notice that we are downloading the BF16 not the FP8 version of the model**. 

#### Run model conversion

Assuming that we have downloaded the HuggingFace checkpoint to a `<SRC_PATH>` directory, we can run [this script](./utils/launch_nemo_convert.sh) (which calls [this python script](./utils/nemo_convert.py)) to perform checkpoint format conversion. After such conversion is done, you should be able to find the converted checkpoint under `<DST_PATH>` directory, and there should be two subfolders inside this directory - `context` and `weights`. 

```bash
# fill in the built container path here
export CONT_IMAGE_URL=""
# fill in the folder that holds the HF checkpoint here
# under this folder, you should see a lot of safetensors
export SRC_PATH=""
# fill in the destination folder of your choice here
# after conversion is done, you can find context and weights under this path
export DST_PATH=""

# Extra Slurm-related arguments can be provided here
sbatch launch_nemo_convert.sh
```

After the model conversion is done, we can then set `MODEL_CKPT=$DST_PATH` together with `FROM_HF=1` when launching our job, so that we can resume training from the converted HF checkpoint. 
