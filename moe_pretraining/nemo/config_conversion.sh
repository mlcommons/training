# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
# Slurm: job time limit
export TIME="02:00:00"
# Slurm: --nodes argument
export NNODES=64
# Slurm: --gpus_per_node and --ntasks_per_node argument
export GPUS_PER_NODE=4

# Folder mapping:
# Output directory for logs
export LOG_DIR="<LOG_DIR>"
# Image / container path, either local cache file or remote URL
export IMAGE="<IMAGE>"

# Conversion settings:
# Path to the input HuggingFace checkpoint on the host
export HF_CKPT="<HF_CKPT>"
# Path to write the converted Megatron-LM checkpoint on the host
export OUTPUT_DIR="<OUTPUT_DIR>"

# Model parallelism (must match the target training configuration):
export TENSOR_PARALLEL_SIZE=1
export PIPELINE_PARALLEL_SIZE=4
export VIRTUAL_PIPELINE_PARALLEL_SIZE=4
export EXPERT_PARALLEL_SIZE=64
