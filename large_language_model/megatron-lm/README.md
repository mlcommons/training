# 1. Problem 
Large Language Model - GPT-3 175B

# 2. Directions

Our codebase is capable of training large language models with both model and data parallelism.

### Steps to configure machine

To use this repository, please install a supported version of PyTorch with GPU support (python 3.8, pytorch 1.12, cuda 11.6.2, and nccl 2.12.10 and above) and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start). We recommend using one of [NGC's PyTorch containers](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch). The latest tested compatible version is `nvcr.io/nvidia/pytorch:22.04-py3`).

### Steps to run and time

To train GPT-3, set `COM_DIR` in `gpt3_blend.sh` to point to the C4 dataset location which contains the dataset after preprocessing.

```
sbatch run_gpt3.sh <path to log directory> <path to BPE processed directory> <container>
```

Use script `run_gpt3.sh` as shown above to run GPT-3 175B on clusters using slurm. You can adjust number of nodes (tested only with nodes>=8) and job run time in the sbatch command in line #3 of the `run_gpt3.sh` script.

Note that the model trains for 15 mins lesser than that actual run time because the last 15 mins are set aside for storing a checkpoint of the last iteration.

Command line arguments are described in detail in this source file [`arguments.py`](./megatron/arguments.py).


# 3. Dataset/Environment

### Background

We use c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4).

For training in the benchmarking region, only 1/4th of the 1024 original `json.gz` files are used. Specifically, the last 1/4th of the files from 768 till 1024 `json.gz` are required.

For validation, a subset of the validation dataset has been selected. This was done by randomly selecting 24,567 examples using [select_example.md](https://github.com/sgpyc/training/blob/paxml-llm-draft/large_language_model/paxml/utils/select_example.md) to get a smaller evaluation dataset.

The dataset is preprocessed using Sentence Piece Model. [These](https://github.com/sgpyc/training/blob/paxml-llm-draft/large_language_model/paxml/utils/generate_spm.md) instructions were used to train the SPM. 

### Data Download
Training dataset -
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
mkdir <${C4_PATH}>
cd <${C4_PATH}>
git lfs pull --include "en/c4-train.007*.json.gz"
git lfs pull --include "en/c4-train.008*.json.gz"
git lfs pull --include "en/c4-train.009*.json.gz"
git lfs pull --include "en/c4-train.01*.json.gz"
```

Validation dataset needs to be downloaded from `gs://mlperf-llm-public2/c4/en_val_subset_json/c4-validation_24567exp.json` to ${C4_PATH}.

### Data Preprocessing for Megatron-LM

Run the following commands to merge these 256 files into 2 `json.gz` files. Each of the `json.gz` files will be preprocessed into a pair of megatron dataset files (`.bin` and `.idx`).

```bash
cd <${C4_PATH}>

# create softlinks to store each shard before merging
mkdir -p softlinks
for shard in {6..7}; do
  start=$((shard * 128))
  end=$((shard * 128 + 127))
  mkdir -p softlinks/en_$shard
  for ind in $(seq -f "%05g" $start $end); do
    ln -s ../../en/c4-train.${ind}-of-01024.json.gz softlinks/en_${shard}/c4-train.${ind}-of-01024.json.gz
  done
done

# merge
mkdir -p en_merge
for shard in {6..7}; do
  cat softlinks/en_${shard}/*gz > en_merge/c4-train.en_${shard}.json.gz 
done
```

After preparing the data folder, download tokenizer model. The tokenizer model should be downloaded from `gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model` and renamed as `${C4_PATH}/tokenizers/c4_spm/sentencepiece.model`. Make sure an output directory `${C4_PATH}/preprocessed_c4_spm` exists before the next step.

Modify `C4_PATH` in `preprocess.sh` and `preprocess_val.sh` to specify
the correct input/output paths and run preprocessing as follows
```bash
cd scripts
sbatch preprocess.sh <path to c4>
sbatch preprocess_val.sh <path to c4> <path to validation json>
```

Currently, the training script expects BPE [vocab.json](https://huggingface.co/gpt2/resolve/main/vocab.json) and [merges.txt](https://huggingface.co/gpt2/resolve/main/merges.txt) files. These files are used to create a BPE tokenizer which is only used for two things at this point in the code since tokenization is already done in the above step:

1. To find out the eod entry index (value is 50256)
2. To find out the vocab size (value is 50257)

Correctness of the dataset preprocessing can be verified by comparing the checksums provided [here](./checksums/dataset_checksum.log)


# 4. Model
### Publication/Attribution
Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf), [2](https://arxiv.org/pdf/2104.04473.pdf), and [3](https://arxiv.org/pdf/2205.05198.pdf)) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA.

### List of Layers

The model largely follows the GPT-3 [paper](https://arxiv.org/abs/2005.14165), Some of the modifications are described below:

1. Tokenizer is changed from BPE to [SentencePiece](https://github.com/google/sentencepiece) with BPE.
2. Alternating sparse attention layers are not used.
3. Model parameters are set [here](https://github.com/mlcommons/training/blob/master/large_language_model/megatron-lm/run_gpt3.sh#L46-L92).

### Model checkpoint
#### Conversion
In the benchmarking region, we should resume training from a PAXML checkpoint which is trained with Global Batch Size of 1536 for 4000 iterations.
Paxml Checkpoint is available at: `gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000`
To resume training from the above checkpoint on Megatron, it should be converted into a format suitable for Megatron (this step only needs to be done once).

To convert Paxml checkpoint to the Megatron's format, a [script](scripts/convert_paxml_to_megatron_distributed.py) has been provided:
```bash
# Convert model and optimizer parameters to Megatron format (runs in ~40 minutes on DGXA100, requires 1TB of CPU memory):
python -u convert_paxml_to_megatron_distributed.py -gckpt $PAXML_CKPT_PATH -o $EXTERNAL_MODEL_CHECKPOINT_DIR --dtype fp32  # or `--dtype bf16` for BF16 checkpoint
# Add framework-specific common.pt file to the checkpoint (instantaneous):
python json_to_torch.py -i common_fp32.json -o $EXTERNAL_MODEL_CHECKPOINT_DIR/common.pt  # or `-i common_bf16.json` for BF16 checkpoint
```
Correctness of the Megatron format checkpoint can be verified by comparing the checksums provided [here](./checksums/fp32_checkpoint_checksum.log). Checksum for two files (`metadata.json` and `common.pt`) may not match. These files are provided [here](./checksums/additional_checkpoint_files) for verification. 

Validation log perplexity can also be used as a metric to verify the correctness of the checkpoint and the loading scripts. To do this, the model should be evaluated on the entire validation dataset after loading weights from the checkpoint. We have observed an average log perplexity of 2.7767 and a standard deviation of 0.00035 (data obtained from 16 runs).

**Note: For BF16 training, the conversion scripts need to be run again with the bf16 arguments specified above**

#### Checkpoint Parameters
There are four groups of parameters in the checkpoint:
1. model FP32 weights (or BF16 weights)
2. first moments of the optimizer state
3. second moments of the optimizer state
4. model FP32 weights copy (created only for BF16 training)

For each model layer we store a separate directory for each of those groups, e.g. for position embeddings:
1. `language_model.embedding.position_embeddings.weight`
2. `optimizer.state.exp_avg.language_model.embedding.position_embeddings.weight` (first moments of the optimizer state)
3. `optimizer.state.exp_avg_sq.language_model.embedding.position_embeddings.weight` (second moments of the optimizer state)
4. `optimizer.state.fp32_from_fp16.language_model.embedding.position_embeddings.weight` (model FP32 weights copy created only for BF16 training)

Each directory contains a single Zarr array (see Zarr section below) and corresponds to a single parameter tensor
(that might be split into different devices during model training).
Pipeline parallel layers are stacked together in a single array.
E.g. for a model with 96 transformer layers, the array corresponding to the self-attention QKV bias
(`language_model.encoder.layers.self_attention.query_key_value.bias`) has shape [**96**, 36864, 12288].

#### Checkpoint Metadata
All non-parameters data is stored in a `common.pt` torch file and contains framework specific information.
An example content of a Megatron specific common.pt file is presented in `scripts/common_bf16.json` file.

Apart from that the checkpoint metadata is stored in `metadata.json` file.

#### Checkpoint Zarr format
Each parameter is stored in a separate directory as a [Zarr](https://zarr.readthedocs.io/) array to allow parallel access.
The content of a single directory is an array fragmented into multiple files (e.g. `0.0`, `0.1`, ...) and should be manipulated
only with Zarr or Zarr-compatible libraries such as [TensorStore](https://google.github.io/tensorstore/).

Megatron features a small library in `megatron.core.dist_checkpointing` that builds on the Zarr and TensorStore primitives
and allows operating on arrays split into different devices (in tensor or pipeline parallel groups).

We recommend to familiarize with the aforementioned libraries, but for convenience
here is a snippet allowing to read a single layer array into a numpy array with either tensorstore or zarr:
```python
import tensorstore as ts
import zarr

def open_with_ts(layer_dir):
    spec = {'driver': 'zarr',
            'metadata_key': '.zarray',
            'kvstore': {'driver': 'file', 'path': layer_dir}}
    return ts.open(ts.Spec(spec), open=True).result().read().result()

def open_with_zarr(layer_dir):
    return zarr.open(layer_dir)[:]

# e.g.
layer_norm_weights_optim_state = open_with_ts('/llm_checkpoint/optimizer.state.exp_avg.language_model.encoder.final_layernorm.weight')
```

Currently NumPy does not support BF16 datatype natively, but it can be added by just importing the tensorstore library (`import tensorstore`).

### How to run
To load an external Megatron format checkpoint (in this case, it is a PAXML checkpoint converted to Megatron format) before training, set the following env variables:
- `EXTERNAL_MODEL_CHECKPOINT_DIR` pointing to the checkpoint directory
- `EXTERNAL_TRAINING_ITERATIONS` to number of iterations the external checkpoint was trained with (default: 4000)
- `EXTERNAL_GBS` to global batch size the external checkpoint was trained with to determine number of samples already consumed (default: 1536)

Note that using an external checkpoint is needed only while training from a checkpoint that was not generated during the current training process in the benchmarking region. When _resuming_ Megatron training (e.g. after hitting a preset node time limit), `EXTERNAL_MODEL_CHECKPOINT_DIR` should not be set.

- Set `USE_BF16` env variable to true for BF16 training.

# 5. Quality

### Quality metric
Log Perplexity

### Quality target
2.69

### Evaluation frequency
Evaluate after every 24576 samples (=50.33B tokens)

### Evaluation thoroughness
Evaluation on the validation subset that consists of 24567 examples.

