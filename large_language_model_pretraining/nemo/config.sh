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
# Slurm: job time limit, defaults to 4 hours
export TIME="04:00:00"
# Slurm: --nodes arguments, default to use 288 nodes
export NNODES=288
# Slurm: --gpus_per_node and --ntasks_per_node argument, defaults to 8 GPUs per node
export GPUS_PER_NODE=8
# Slurm: max job retries for transient job failures, defaults to retry 3 times
export MAX_RETRIES=3

# Folder mapping:
# Output directory that holds logs, any path that you like. 
export JOB_DIR=""
# Image / container path, either local cache file or remote URL
export IMAGE=""
# Dataset: C4 dataset location that contains the dataset after preprocessing
# This corresponds to the PREPROCESSED_PATH in README section 3's dataset download part
export PREPROCESSED_PATH=""
# Dataset: Numpy index working directory, contains shuffled dataset
# This path must be able to hold >400GB data
export TMP_NPY_INDEX=""
# Dataset: Tokenizer path
# This corresponds to the TOKENIZER_PATH in README section 3's tokenizer download part
export TOKENIZER_PATH=""

# Model: checkpoint and tokenizer path
#     This is the checkpoint that we want to start with. 
#     Each checkpoint should be a folder containing two sub-folders: context and weights. 
#     And we need to pass this folder's path (the folder containing context and weights) here.  
export MODEL_CKPT=""
# Model: Continual checkpoint directory to write and resume
#     This is the directory to hold all intermediate checkpoints. 
#     Once a run is complete and we specify to save checkpoints, 
#     we should see a checkpoint written in this folder
#     with name `checkpoint-par-x-y-steps`
#     Inside this directory, there should be a `checkpoint` directory that holds context and weights
#     which is the "actual checkpoint". 
#     Notice that this path must be able to hold at least 5.2TB data since each checkpoint is 5.2TB. 
export CONTINUAL_CKPT=""
# Model: Whether we want to restore from MODEL_CKPT path. If 0, then we are not restoring. 
export USE_CKPT=0
# Model: Whether we are resuming from a NeMo-formatted HuggingFace checkpoint (weights only). 
#     If set to 1, then checkpoint resuming code will not try to load the optimizer states. 
export FROM_HF=1
# Model: Whether we want to save a checkpoint. Must be 1 if NPAR > 1. If 1, then we save a checkpoint at the end.
export SAVE_CKPT=0


# Training Configs: 
# Model: size, to choose from 8b, 70b, 405b
export SIZE="405b"
# Dataloader: Global batch size
export GBS=1152
# Dataloader: Micro batch size
export MBS=1
# Dataloader: Max run N batches, optional
#     If an empty string is provided (""), then the training will continue until time limit
#     If we want to save a checkpoint, then this value must be set
export MAX_STEPS=""

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
export SEEDS=""