#!/bin/bash

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

set -e

git config --global --add safe.directory /workspace/llama31

# Vars without defaults
#     Slurm settings
: "${USER:?USER not set}"
: "${HOST:?HOST not set}"
: "${ACCOUNT:?ACCOUNT not set}"
: "${PARTITION:?PARTITION not set}"

#     Job settings
: "${JOB_DIR:?JOB_DIR not set}"
: "${IMAGE:?IMAGE not set}"

#     Dataset settings
: "${PREPROCESSED_DATA:?PREPROCESSED_DATA not set}"
: "${SPM_CKPT:?SPM_CKPT not set}"


# Vars with defaults
#     Slurm settings
: "${TIME:="00:30:00"}"
: "${NNODES:=72}"
: "${GPUS_PER_NODE:=8}"
: "${DEPENDENCIES:=""}"

#     Job settings
: "${NEMO_DIR:=""}" # Provide customized NeMo path here
: "${TMP_NPY_INDEX:=""}" # Provide temporary NNumpy Index saving directory

#     Model settings
: "${SIZE:="405b"}"
: "${GBS:=288}"
: "${MBS:=1}"


# Run

MOUNTS="${JOB_DIR}:/output,${PREPROCESSED_DATA}:/preproc_data,${SPM_CKPT}:/workspace/llm/tokenizer.model"

if [ ! $NEMO_DIR = "" ]; then
    MOUNTS="${MOUNTS},${NEMO_DIR}:/opt/NeMo"
fi

if [ ! $TMP_NPY_INDEX = "" ]; then
    MOUNTS="${MOUNTS},${TMP_NPY_INDEX}:/npy_index"
fi

if [ ! $DEPENDENCIES = "" ]; then 
    DEPENDENCIES="--dependencies ${DEPENDENCIES}"
fi

set -x

python3 pretrain_llama31.py \
--user $USER --host $HOST \
--job_dir $JOB_DIR \
--account $ACCOUNT --partition $PARTITION \
--nodes $NNODES --gpus_per_node $GPUS_PER_NODE \
$DEPENDENCIES \
--time $TIME \
--mounts $MOUNTS \
--image $IMAGE \
--size $SIZE \
--gbs $GBS --mbs $MBS
