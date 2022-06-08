# 1. Problem 
Large Language Model - GPT-3 175B
# 2. Directions

Our codebase is capable of training large language models with both model and data parallelism
.
To use this repository, please install a supported version of PyTorch with GPU support (python 3.8, pytorch 1.8, cuda 11.1, and nccl 2.8.3 and above) and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start). We recommend using one of [NGC's PyTorch containers](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch). The latest tested compatible version is `nvcr.io/nvidia/pytorch:21.12-py3`).

To train GPT-3, set `COM_DIR` in `gpt3_blend.sh` to point to the C4 dataset location which contains the dataset after preprocessing.

```
sbatch run_gpt3.sh <path to log directory> <path to BPE processed directory>
```

Use script `run_gpt3.sh` as shown above to run GPT-3 175B on clusters using slurm. You can adjust number of nodes (tested only with nodes>=8) and job run time in the sbatch command in line #3 of the `run_gpt3.sh` script.

Some things to note regarding the above script -
1. Currently the model computes validation on a subset of the training dataset and the data has been split as `98,2,0` right now.
2. The model trains for 15 mins lesser than that actual run time because the last 15 mins are set aside for storing a checkpoint of the last iteration.
3. The tokenization scheme used is currently BPE (which requires a `merge table` and a `json` vocabulary file). Scripts for data preprocessing using SenetencePiece tokenizer will be added in the future.

Command line arguments are described in detail in this source file [`arguments.py`](./megatron/arguments.py).


# 3. Dataset/Environment
We use C4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4).
We do not host any datasets for GPT training.
# 4. Model
### Publication/Attribution
Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf) and [2](https://arxiv.org/pdf/2104.04473.pdf)) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA.
