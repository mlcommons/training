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

# Folder mapping:
# Output directory that holds logs
export JOB_DIR=""
# Image path, either local cache file or remote URL
export IMAGE=""
# Dataset: C4 dataset location that contains the dataset after preprocessing
export PREPROCESSED_DATA=""
# Dataset: Trained SentencePieceModel checkpoint path
export SPM_CKPT=""

# Optional
# Numpy index working directory
export TMP_NPY_INDEX=""

# Fixed
# Model: size, to choose from 8b, 70b, 405b
export SIZE=""
# Dataloader: Global batch size
export GBS=0
# Dataloader: Micro batch size
export MBS=0