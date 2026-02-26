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

# SLURM: Username on a cluster
export USER="<SLURM_USER>"
# Slurm: account for job submission 
export ACCOUNT="<SLURM_ACCOUNT>"
# Slurm: partition for job submission
export PARTITION="<SLURM_PARTITION>"
# Slurm: job time limit, defaults to 8 hours
export TIME="02:00:00"
# Slurm: --nodes arguments
export NNODES=64
# Slurm: --gpus_per_node and --ntasks_per_node argument
export GPUS_PER_NODE=4
# Slurm: max job retries for transient job failures
export MAX_RETRIES=1

# Folder mapping:
# Output directory that holds logs, any path that you like. 
export LOG_DIR="<LOG_DIR>"
# Image / container path, either local cache file or remote URL
export IMAGE="<IMAGE>"
# Dataset: C4 dataset location that contains the dataset after preprocessing
export DATA_DIR="<DATA_DIR>"
# Model checkpoint path
export MODEL_CKPT="<MODEL_CKPT>"

# Training Configs: 
# Dataloader: Global batch size
export GBS=16384
# Dataloader: Micro batch size
export MBS=1
export MAX_LR=0.000024
export MIN_LR=1e-8
export MAX_STEPS=12000
export WARMUP_STEPS=4
export EVAL_CHECK_INTERVAL=1  # every $EVAL_CHECK_INTERVAL steps
export EVAL_BATCHES=1  # evaluate on $EVAL_BATCHES * $GBS samples
export EVAL_BATCH_SIZE=1024


export TENSOR_PARALLEL_SIZE=1
export PIPELINE_PARALLEL_SIZE=4
export CONTEXT_PARALLEL_SIZE=1
export EXPERT_PARALLEL_SIZE=64
export EXPERT_TENSOR_PARALLEL_SIZE=1
export RECOMPUTE_MODULES="mlp,moe_act"

# Experiment manager: Number of experiments to launch
export NEXP=1
# Experiment manager: how many consecutive jobs we want for each experiment
export NPAR=1
# Experiment manager: provides seeds to the launched experiments, use space as delimiter, such as "1234 1235 1236"
#     The training script will discard all excessive seeds, and generate seeds if given seeds < NEXP. 
#     To preserve randomness, we recommend not to set this value so that each time seeds can be randomly generated. 


export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
