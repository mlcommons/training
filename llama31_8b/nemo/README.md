# 1. Problem 

Small Language Model pretraining - Llama 3.1 8B

# 2. Docker Setup
To build the docker image: 
```bash
docker build -t <image-tag> -f Dockerfile .
```

To launch the docker container: 
```
docker run -it --rm \
    --net=host --uts=host \
    <image-tag>
```


# 3. Dataset and Model

The current codebase is using the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4) for train and evaluation. 

## Preprocessed data download

The pre-tokenized dataset and the tokenizer are available to download. More information about the MLCommons R2 Downloader, including how to run it on Windows, are available [here](https://training.mlcommons-storage.org). You can download using the following commands:

```bash
# data 
# go to the path where you want the data to be downloaded
# use the same path in config when exporting PREPROCESSED_PATH
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d llama3_1_8b_preprocessed_c4_dataset https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
```
```bash
# tokenizer 
# go to the path where you want the tokenizer to be downloaded
# use the same path in config when exporting TOKENIZER_PATH
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d llama3_1_8b_tokenizer https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
```

## Raw data downloading [Optional]

We use [AllenAI C4](https://huggingface.co/datasets/allenai/c4) dataset for this benchmark. The original zipped **`json.gz`** files can be downloaded by following AllenAI C4's instruction, and you can download our zipped customized validation dataset from the MLCommons S3 bucket by running the following command: 


```bash
export C4_PATH=""

# download the full C4 files, including all raw train and validations
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d $C4_PATH https://training.mlcommons-storage.org/metadata/c4-full-dataset-unzipped.uri
```
After downloading, run the following command to process them to zip them into `.gz` format before running the data preprocessing. 

```bash
bash utils/parallel_compress_json_to_gz.sh
```

Run the following commands to merge all 1024 training files into 8 `json.gz` files, all 8 validation files into a single `json.gz` file, as well as generate our customized validation dataset. Each of the `json.gz` files will subsequently be preprocessed into a pair of megatron dataset files (`.bin` and `.idx`) by our preprocess.sh script. 

```bash
export C4_PATH=""
export MERGED_C4_PATH=""
# more information about this knob can be found in consolidate_data.sh
export N_VALIDATION_SAMPLES=91205

bash utils/consolidate_data.sh
```

### Tokenizer
We are using the Llama 3.1 8B tokenizer. To download it, you can run the following commands:
```bash
export TOKENIZER_PATH=""
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B  --local-dir $TOKENIZER_PATH
```

After the data consolidation is done, we can perform preprocessing using the following commands: 

```bash
# pass in the folder path that contains the Llama tokenizer here
# please refer to the tokenizer section above for more details
export TOKENIZER_PATH=""
# pass in the merged file path here
export MERGED_C4_PATH=""
# this path is used for storing the preprocessed .bin and .idx files
export PREPROCESSED_PATH=""

for index in {0..7}; do
    # please specify the right path to nemo
    python3 </path/to/nemo>/nemo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input "${MERGED_C4_PATH}/c4-train.en_${index}.json.gz" \
    --output-prefix "${PREPROCESSED_PATH}/c4-train.en_${index}" \
    --tokenizer-library huggingface --tokenizer-type ${TOKENIZER_PATH} \
    --dataset-impl mmap --workers 128 &
done
    # please specify the right path to nemo
    python3 </path/to/nemo>/nemo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input "${MERGED_C4_PATH}/c4-validation-91205-samples.en.json.gz" \
    --output-prefix "${PREPROCESSED_PATH}/c4-validation-91205-samples.en" \
    --tokenizer-library huggingface --tokenizer-type ${TOKENIZER_PATH} \
    --dataset-impl mmap --workers 128 & 
wait

```

After the download is complete, you should see files with the following naming conventions under `PREPROCESSED_PATH`, ending with both `.idx` and `.bin`: 
- Training partitions: `c4-train.en_<number>_text_document`
- Validation partitions: `c4-validation-91205-samples.en_text_document`

#### Training and test data separation

We use the default split from the C4 dataset. This means that we use `c4-train.<x>-of-01024.json.gz` files (where `768 <= x <= 1023`) for training, and we use our customized `c4-validation-91205-samples.en.json.gz`, which contains the first 91205 samples from the unshuffled C4 validation dataset, for evaluation. 

Notice here that we are using the first 1024 sequences (8,388,608 tokens) from the validation dataset to perform the validation. According to our experiments, the first 91205 samples from the unshuffled C4 dataset yields 47,186,855 tokens, which is the smallest amount of samples needed to yield 47,185,920 tokens. Thus, we have chosen the first 91205 samples as our validation dataset. 

#### Training data order

We randomly shuffle the **last 256 of 1024 shards** for the benchmarking area.

#### Test data order

We use the first 1024 sequences in the validation dataset for validation. We **do not shuffle** the validation dataset. 

# 4. Model
### Publication/Attribution

The model largely follows the Llama 3.1 8B [paper](https://arxiv.org/abs/2407.21783). 

### Model details

| Config | Value |
| :-- | :-- | 
| Embedding | RoPE + parameter adjustments |
| # Layers | 32 | 
| Attention Type | GQA |
| # Attn Heads | 32 | 
| Key/Value Heads | 8 | 
| Model Dimension | 4096 |
| FFN Dimension | 14336 |
| Activation | SwiGLU | 
| Normalization | RMSNorm |  
| Tokenizer | Llama tokenizer |
| Vocab size | 128,000 |  
| Context Length | 8192 |


#### Saving and restoring a checkpoint

Large runs might need to span across multiple Slurm jobs, and we need to save and load checkpoints with contexts so that training can resume between jobs. To support this, we have added some environment variables. Please refer to `config.sh` for more details. 

### Optimizer spec

1. Optimizer type: **AdamW**
2. Warmup steps computed as 10% of the total allocated steps.

# 5. Quality
### Quality metric

Validation loss

### Quality target

Validation log perplexity = 3.3

### Evaluation frequency

We perform evaluation every **12288** sequences. 

### Evaluation thoroughness

We evaluate using **1024** sequences from our customized validation dataset. 

# 6. Launch a training run

To train Llama 3.1 8B, we need to fill out all fields in `config.sh`. This file contains all configurations for Slurm cluster access and job submission configurations, directory mappings, containers, and model configurations. 

Once the `config.sh` is properly filled, we launch a training run using the following commands:

```bash
source config.sh
bash run_llama31.sh
```