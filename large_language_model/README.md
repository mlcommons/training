# 1. Problem 
Large Language Model - GPT-3 175B
# 2. Directions

Our codebase is capable of training large language models with both model and data parallelism
.
To use this repository, please install a supported version of PyTorch with GPU support (python 3.8, pytorch 1.8, cuda 11.1, and nccl 2.8.3 and above) and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start). We recommend using one of [NGC's PyTorch containers](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch). The latest tested compatible version is `nvcr.io/nvidia/pytorch:22.04-py3`).

To train GPT-3, set `COM_DIR` in `gpt3_blend.sh` to point to the C4 dataset location which contains the dataset after preprocessing.

```
sbatch run_gpt3.sh <path to log directory> <path to BPE processed directory> <container>
```

Use script `run_gpt3.sh` as shown above to run GPT-3 175B on clusters using slurm. You can adjust number of nodes (tested only with nodes>=8) and job run time in the sbatch command in line #3 of the `run_gpt3.sh` script.

Some things to note regarding the above script -
1. The model trains for 15 mins lesser than that actual run time because the last 15 mins are set aside for storing a checkpoint of the last iteration.

Command line arguments are described in detail in this source file [`arguments.py`](./megatron/arguments.py).


# 3. Dataset/Environment
We use C4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4).
We do not host any datasets for GPT training.
For validation, a subset of the validation dataset has been selected. Details as follows:
24,567 examples were [selected](https://github.com/sgpyc/training/blob/paxml-llm-draft/large_language_model/paxml/utils/select_example.md) in the validation split to form a smaller eval set. The resulting tfrecord file is at gs://mlperf-llm-public2/c4/en/3.0.1/c4-validation_24567exp.tfrecord , with hashes of the text at gs://mlperf-llm-public2/c4/en/3.0.1/c4-validation_24567exp.hash.

### Data Preprocessing using SPM
Benchmarking region will use 1/4th of the 1024 original `json.gz` files, from 768 till 1024 `json.gz` files
Run the following commands to merge these files into 2 `json.gz` files. Each of the 2 `json.gz` file will be preprocessed into a pair of megatron dataset files (`.bin` and `.idx`).

```bash
cd <path to C4>

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
cat en/c4-validation.0000* > en_merge/c4-validation.json.gz
```

After preparing the data folder, download tokenizer model.
Currently, SPM trained by google using [these](https://github.com/sgpyc/training/blob/paxml-llm-draft/large_language_model/paxml/utils/generate_spm.md) instructions is used.

Modify `C4_PATH` in `preprocess.sh` and `preprocess_val.sh` to specify
the correct input/output paths and run preprocessing as follows
```bash
sbatch preprocess.sh <path to c4>
sbatch preprocess_val.sh <path to c4>
```

Currently, the training script expects BPE [vocab.json](https://huggingface.co/gpt2/resolve/main/vocab.json) and [merges.txt](https://huggingface.co/gpt2/resolve/main/merges.txt) files. These files are used to create a BPE tokenizer which is only used for two things at this point in the code since tokenization is already done in the above step:

1. To find out the eod entry index (value is 5025)
2. To find out the vocab size (value is 50257)

Correctness of the dataset preprocessing can be verified by comparing the checksums provided [here](./checksums/dataset_checksum.log)

# 3. External Checkpoints

For the benchmarking region, we would be resuming training from a PAXML checkpoint which is trained on the Global Batch Size of 1536 for 4000 iterations.
Paxml Checkpoint is available at: gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000
To resume training from the above checkpoint on Megatron, we would need to first convert the above checkpoint into a format suitable for Megatron.
This step only needs to be done once.

### Paxml checkpoints conversion
To convert Paxml checkpoint to the Megatron's format, a [script](scripts/convert_paxml_to_megatron_distributed.py) has been provided:
```bash
# Convert model and optimizer parameters to Megatron format (runs in ~40 minutes on DGXA100, requires 1TB of CPU memory):
python -u convert_paxml_to_megatron_distributed.py -gckpt $PAXML_CKPT_PATH -o $EXTERNAL_MODEL_CHECKPOINT_DIR --dtype bf16  # or `--dtype fp32` for FP32 checkpoint
# Add framework-specific common.pt file to the checkpoint (instantaneous):
python json_to_torch.py -i common_bf16.json -o $EXTERNAL_MODEL_CHECKPOINT_DIR/common.pt  # or `-i common_fp32.json` for FP32 checkpoint
```

### How to run
To run external checkpoints (including PAXML checkpoint converted to Megatron compliant format), set the following env variables:
- `EXTERNAL_MODEL_CHECKPOINT_DIR` pointing to the checkpoint directory
- `EXTERNAL_TRAINING_ITERATIONS` to number of iterations the external checkpoint was trained with (default: 4000)
- `EXTERNAL_GBS` to global batch size the external checkpoint was trained with to determine number of samples already consumed (default: 1536)

Note that using an external checkpoint is needed only for the first training run. When _resuming_ Megatron training (e.g. after hitting a time limit), `EXTERNAL_MODEL_CHECKPOINT_DIR` should not be set.

# 4. Model
### Publication/Attribution
Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf), [2](https://arxiv.org/pdf/2104.04473.pdf), and [3](https://arxiv.org/pdf/2205.05198.pdf)) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA.
