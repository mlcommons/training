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
export USER=""
# SSH: remote cluster URL
export HOST=""
# Slurm: account for job submission 
export ACCOUNT=""
# Slurm: partition for job submission
export PARTITION=""
# Slurm: job time limit
export TIME=""
# Slurm: --nodes arguments 
export NNODES=0
# Slurm: --gpus_per_node and --ntasks_per_node argument
export GPUS_PER_NODE=0
# Slurm: max job retries for transient job failures
export MAX_RETRIES=0

# Folder mapping:
# Output directory that holds logs
export JOB_DIR=""
# Image path, either local cache file or remote URL
export IMAGE=""
# Dataset: C4 dataset location that contains the dataset after preprocessing
export PREPROCESSED_PATH=""
# Dataset: Numpy index working directory
export TMP_NPY_INDEX=""
# Dataset: Tokenizer path
export TOKENIZER_PATH=""

# Environment: NeMo remount
export NEMO_DIR=""

# Model: checkpoint and tokenizer path
#     This is the checkpoint that we want to start with. 
#     Each checkpoint should be a folder containing two sub-folders: context and weights. 
#     And we need to pass this folder's path (the folder containing these two sub-folders) here.  
export MODEL_CKPT=""
# Model: Continual checkpoint directory to write and resume
#     This is the directory to hold all intermediate checkpoints. 
#     Once a run is complete and we specify to save checkpoints, 
#     we should see a checkpoint written in this folder
#     with name `checkpoint-par-x-y-steps`
#     Inside this directory, there should be a `checkpoint` directory that holds context and weights
#     which is the "actual checkpoint"
export CONTINUAL_CKPT=""
# Model: Whether we want to restore from MODEL_CKPT path. If 0, then we are not restoring. 
export USE_CKPT=0
# Model: Whether we want to save a checkpoint. Must be true if NPAR > 1
export SAVE_CKPT=0


# Training Configs: 
# Model: size, to choose from 8b, 70b, 405b
export SIZE=""
# Dataloader: Global batch size
export GBS=0
# Dataloader: Micro batch size
export MBS=0
# Dataloader: Evaluate every N batches, optional
export EVAL_EVERY=""
# Dataloader: Evaluate using N batches, optional
export EVAL_BATCHES=""
# Dataloader: Max run N batches, optional
export MAX_STEPS=""

# Experiment: starting steps
#     This is the starting "offset" step from the checkpoint. 
#     For instance, if you are resuming from a checkpoint folder `checkpoint-par-x-y-steps/checkpoint`, 
#     then the value y is needed here. 
export START_STEPS=""
# Experiment manager: Number of experiments to launch
export NEXP=0
# Experiment manager: how many consecutive jobs we want for each experiment
export NPAR=0
# Experiment manager: provides seeds to the launched experiments, use space as delimiter, such as "1234 1235 1236"
# The training script will discard all excessive seeds, and generate seeds if given seeds < NEXP. 
export SEEDS=""