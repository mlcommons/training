# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SSH: username that connects to the remote cluster
export USER="DUMMY"
# SSH: remote cluster URL
export HOST="DUMMY"
# Slurm: account for job submission 
export ACCOUNT="DUMMY"
# Slurm: partition for job submission
export PARTITION="DUMMY"
# Slurm: job time limit, defaults to 8 hours
export TIME="08:00:00"
# Slurm: --nodes arguments, default to use 288 nodes
export NNODES=1
# Slurm: --gpus_per_node and --ntasks_per_node argument, defaults to 8 GPUs per node
export GPUS_PER_NODE=8
# Slurm: max job retries for transient job failures, defaults to retry 3 times
export MAX_RETRIES=1

# Folder mapping:
# Output directory that holds logs, any path that you like. 
export JOB_DIR="/workspace/code/logs"
# Image / container path, either local cache file or remote URL
export IMAGE="DUMMY"
# Dataset: C4 dataset location that contains the dataset after preprocessing
# export ORIGINAL_C4_PATH="/data/data/C4"

# This corresponds to the PREPROCESSED_PATH in README section 3's dataset download part
export PREPROCESSED_PATH="/data/llama31_8b/data/C4_processed"
export MERGED_C4_PATH="/data/llama31_8b/data/C4_merged"
# Dataset: Numpy index working directory, contains shuffled dataset
# This path must be able to hold >400GB data
export TMP_NPY_INDEX="/data/npy_indices"
# Dataset: Tokenizer path
# This corresponds to the TOKENIZER_PATH in README section 3's tokenizer download part
export TOKENIZER_PATH="/data/llama31_8b/model/Llama-3.1-8B-ref/"
# export TOKENIZER_PATH="/data/llama3_405b_ref/tokenizer"

# Model: Continual checkpoint directory to write and resume
#     This is the directory to hold all intermediate checkpoints. 
#     Once a run is complete and we specify to save checkpoints, 
#     we should see a checkpoint written in this folder
#     with name `checkpoint-par-x-y-steps`
#     Inside this directory, there should be a `checkpoint` directory that holds context and weights
#     which is the "actual checkpoint". 
#     Notice that this path must be able to hold at least 5.2TB data since each checkpoint is 5.2TB. 
export CONTINUAL_CKPT="/data/model/saved_ckpts"
# Model: Whether we want to restore from MODEL_CKPT path. If 0, then we are not restoring. 
export USE_CKPT=0
# Model: Whether we are resuming from a NeMo-formatted HuggingFace checkpoint (weights only). 
#     If set to 1, then checkpoint resuming code will not try to load the optimizer states. 
export FROM_HF=1
# Model: Whether we want to save a checkpoint. Must be 1 if NPAR > 1. If 1, then we save a checkpoint at the end.
export SAVE_CKPT=0

# Training Configs: 
# Model: size, to choose from 8b, 70b, 405b
export SIZE="8b"
# Dataloader: Global batch size
export GBS=32
# Dataloader: Micro batch size
export MBS=4
export MAX_LR="1e-3"
# Dataloader: Max run N batches, optional
#     If an empty string is provided (""), then the training will continue until time limit
#     If we want to save a checkpoint, then this value must be set
# Fixed max_steps=1200000 in pretrain_llama31.py  
export WARMUP_STEPS=512 # 16384 // GBS
export EVAL_EVERY=12288
export START_EVAL_AT=0

export TENSOR_PARALLEL_SIZE=1
# Experiment: starting steps
#     This is the starting "offset" step from the checkpoint. 
#     For instance, if you are resuming from a checkpoint folder `checkpoint-par-0-20-steps/checkpoint`, 
#     which means that the model is trained for 20 steps to generate the checkpoint, 
#     then the value 20 is needed here. 
export START_STEPS="0"
# Experiment manager: Number of experiments to launch
export NEXP=1
# Experiment manager: how many consecutive jobs we want for each experiment
export NPAR=1
# Experiment manager: provides seeds to the launched experiments, use space as delimiter, such as "1234 1235 1236"
#     The training script will discard all excessive seeds, and generate seeds if given seeds < NEXP. 
#     To preserve randomness, we recommend not to set this value so that each time seeds can be randomly generated. 
# export SEEDS=7963
# export SEEDS=4786

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# Extra configs 
export NCCL_MIN_P2P_NCHANNELS=32;
export NCCL_MIN_CTAS=32;
export NCCL_NCHANNELS_PER_NET_PEER=32;
export NCCL_NVLS_ENABLE=0

export TP_COMM_OVERLAP=False 
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export MC_TP_OVERLAP_RS_DGRAD=False

export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE

export NVTE_RS_STRIDED_ATOMIC=2
export NVTE_FP8_DPA_BWD=1
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_USE_HIPBLASLT=1
export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export USE_TE_SWIGLU=1

# FAv3
export NVTE_CK_USES_BWD_V3=1        # enable dqdkdv bwd kernel
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0  # 16bit atomics

# ck logging
export CK_FUSED_ATTN_LOG_CONFIG=0   # Diable logging for CK fused attn. Enabled for debugging onl

export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

export FUSED_SOFTMAX=0
export RMSNORM_CAST=0

export PT_TENSOR_VALIDATION=0
export PROFILE_RPD=0

export USE_HIPBLASLT=1
export TORCH_BLAS_PREFER_HIPBLASLT=1

export NVTE_USE_RMSNORM_TRITON=1
export ENABLE_TRANSPOSE_CACHE=0
