# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
export NNODES=64
# Slurm: --gpus_per_node and --ntasks_per_node argument, defaults to 8 GPUs per node
export GPUS_PER_NODE=4
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
export PREPROCESSED_PATH="/data/deepseek_v3_671b/data/C4_processed"
export MERGED_C4_PATH="/data/deepseek_v3_671b/data/C4_merged"
# Dataset: Numpy index working directory, contains shuffled dataset
# This path must be able to hold >400GB data
export TMP_NPY_INDEX="/data/npy_indices"
# Dataset: Tokenizer path
# This corresponds to the TOKENIZER_PATH in README section 3's tokenizer download part
export TOKENIZER_PATH="/data/deepseek_v3_671b/model/DeepSeek-V3-671B-Base"
# export TOKENIZER_PATH="/data/llama3_405b_ref/tokenizer"

export MODEL_CKPT="$TOKENIZER_PATH"

# Training Configs: 
# Dataloader: Global batch size
export GBS=1024
# Dataloader: Micro batch size
export MBS=1
export MAX_LR="2e-4"
export WARMUP_STEPS=256
export EVAL_CHECK_INTERVAL=10  # every $EVAL_CHECK_INTERVAL steps
export EVAL_BATCHES=1  # evaluate on $EVAL_BATCHES * $GBS samples


export TENSOR_PARALLEL_SIZE=1
export PIPELINE_PARALLEL_SIZE=4
export CONTEXT_PARALLEL_SIZE=1
export EXPERT_PARALLEL_SIZE=64
export EXPERT_TENSOR_PARALLEL_SIZE=1
export RECOMPUTE_MODULES="mlp,moe_act"
export CUDA_GRAPH_IMPLEMENTATION="transformer_engine"
export CUDA_GRAPH_SCOPE="attn"

# Experiment manager: Number of experiments to launch
export NEXP=1
# Experiment manager: how many consecutive jobs we want for each experiment
export NPAR=1
# Experiment manager: provides seeds to the launched experiments, use space as delimiter, such as "1234 1235 1236"
#     The training script will discard all excessive seeds, and generate seeds if given seeds < NEXP. 
#     To preserve randomness, we recommend not to set this value so that each time seeds can be randomly generated. 


export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
