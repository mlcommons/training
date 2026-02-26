#!/bin/bash

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

cd $(dirname $0)

set -e

# Vars without defaults
: "${ACCOUNT:?ACCOUNT not set}"
: "${PARTITION:?PARTITION not set}"
: "${LOG_DIR:?LOG_DIR not set}"
: "${IMAGE:?IMAGE not set}"
: "${HF_CKPT:?HF_CKPT not set}"
: "${OUTPUT_DIR:?OUTPUT_DIR not set}"

# Vars with defaults
: "${TIME:="02:00:00"}"
: "${NNODES:=64}"
: "${GPUS_PER_NODE:=4}"
: "${GPU:="gb300"}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${PIPELINE_PARALLEL_SIZE:=4}"
: "${VIRTUAL_PIPELINE_PARALLEL_SIZE:=4}"
: "${EXPERT_PARALLEL_SIZE:=64}"
: "${DRYRUN:=0}"
: "${DETACH:=1}"

# Build mounts
MOUNTS="${HF_CKPT}:/input_checkpoint"
MOUNTS="${MOUNTS},${OUTPUT_DIR}:/output_checkpoint"

# Build launcher arguments
LAUNCHER_ARGS="--account $ACCOUNT --partition $PARTITION"
LAUNCHER_ARGS="$LAUNCHER_ARGS --nodes $NNODES --gpus_per_node $GPUS_PER_NODE"
LAUNCHER_ARGS="$LAUNCHER_ARGS --gpu $GPU"
LAUNCHER_ARGS="$LAUNCHER_ARGS --time_limit $TIME"
LAUNCHER_ARGS="$LAUNCHER_ARGS --container_image $IMAGE"
LAUNCHER_ARGS="$LAUNCHER_ARGS --log_dir $LOG_DIR"
LAUNCHER_ARGS="$LAUNCHER_ARGS --mounts $MOUNTS"

if [ "$DRYRUN" -gt 0 ]; then
    LAUNCHER_ARGS="$LAUNCHER_ARGS --dryrun"
fi

if [ "$DETACH" -gt 0 ]; then
    LAUNCHER_ARGS="$LAUNCHER_ARGS --detach"
fi

# Build conversion arguments
CONVERT_ARGS="--hf-model-id /input_checkpoint"
CONVERT_ARGS="$CONVERT_ARGS --output-dir /output_checkpoint"
CONVERT_ARGS="$CONVERT_ARGS --tp $TENSOR_PARALLEL_SIZE"
CONVERT_ARGS="$CONVERT_ARGS --pp $PIPELINE_PARALLEL_SIZE"
CONVERT_ARGS="$CONVERT_ARGS --vp $VIRTUAL_PIPELINE_PARALLEL_SIZE"
CONVERT_ARGS="$CONVERT_ARGS --ep $EXPERT_PARALLEL_SIZE"
CONVERT_ARGS="$CONVERT_ARGS --trust-remote-code"

# Allows MLLogger objects to be constructed locally
if [ ! -d /mlperf-outputs ]; then mkdir -p /mlperf-outputs 2>/dev/null || true; fi

set -x

python3 run_conversion.py \
    $LAUNCHER_ARGS \
    -- \
    $CONVERT_ARGS
