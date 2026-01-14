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

#git config --global --add safe.directory /workspace/llama31

# Vars without defaults
#     Slurm settings
: "${USER:?USER not set}"
: "${HOST:?HOST not set}"
: "${ACCOUNT:?ACCOUNT not set}"
: "${PARTITION:?PARTITION not set}"
: "${REMOTE:=0}"

#     Job settings
: "${JOB_DIR:?JOB_DIR not set}"
: "${IMAGE:?IMAGE not set}"

#     Dataset settings
: "${PREPROCESSED_PATH:?PREPROCESSED_PATH not set}"
: "${TOKENIZER_PATH:?TOKENIZER_PATH not set}"

# Vars with defaults
#     Slurm settings
: "${TIME:="04:00:00"}"
: "${NNODES:=1}"
: "${GPUS_PER_NODE:=8}"
: "${DEPENDENCIES:=""}"

#     Job settings
: "${NEMO_RUN_DIR:=""}" # Provide customized NeMo-Run path here
: "${TMP_NPY_INDEX:=""}" # Provide temporary NNumpy Index saving directory
: "${MAX_RETRIES:=0}"

#     Model settings
: "${GBS:=1024}"
: "${MBS:=1}"

# Eval settings
: "${EVAL_CHECK_INTERVAL:=10}"
: "${EVAL_BATCHES:=1}"

#     Dataloader settings
: "${MAX_STEPS:="1200000"}"

#     Experiment settings
: "${SEEDS:=""}"
IFS=" " read -ra seeds <<< $SEEDS
: "${NEXP:=1}"
: "${NPAR:=1}"
: "${TAG:=""}"
: "${TARGET:="1.0"}"  # TODO(dfridman): update once determined
: "${STEP_TIME_ATOL:="18000"}" # maximum tolerable step time, setting to 5hr by default

# Run

MOUNTS="${JOB_DIR}:/output,${JOB_DIR}:/mlperf-outputs,${PREPROCESSED_PATH}:/preproc_data,${TOKENIZER_PATH}:/tokenizer,${MODEL_CKPT}:/checkpoint"

CMD_SUFFIX=""

if [ ! $NEMO_RUN_DIR = "" ]; then
    MOUNTS="${MOUNTS},${NEMO_RUN_DIR}:/opt/NeMo-Run"
fi

if [ ! $TMP_NPY_INDEX = "" ]; then
    MOUNTS="${MOUNTS},${TMP_NPY_INDEX}:/npy_index"
fi

if [ ! $DEPENDENCIES = "" ]; then 
    CMD_SUFFIX="${CMD_SUFFIX} --dependencies ${DEPENDENCIES}"
fi

if [ ! $MAX_STEPS = "" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --max_steps ${MAX_STEPS}"
fi

if [ ! $TAG = "" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --tag ${TAG}"
fi

if [ $REMOTE -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --run_slurm"
fi

# Allows MLLogger objects to be constructed locally
if [ ! -d /mlperf-outputs ]; then mkdir /mlperf-outputs; fi

set -x

python3 pretrain_llama31.py \
--user $USER --host $HOST \
--job_dir $JOB_DIR \
--account $ACCOUNT --partition $PARTITION \
--nodes $NNODES --gpus_per_node $GPUS_PER_NODE \
--time $TIME \
--max_retries $MAX_RETRIES \
--mounts $MOUNTS \
--image $IMAGE \
--size $SIZE \
--gbs $GBS \
--mbs $MBS \
--seeds ${seeds[@]} \
--num_exps $NEXP \
--num_pars $NPAR \
--tokenizer_path $TOKENIZER_PATH \
--target_log_ppl $TARGET \
--step_time_atol $STEP_TIME_ATOL \
--warmup_steps $WARMUP_STEPS \
--tensor_parallel_size $TENSOR_PARALLEL_SIZE \
--pipeline_parallel_size $PIPELINE_PARALLEL_SIZE \
--context_parallel_size $CONTEXT_PARALLEL_SIZE \
--expert_model_parallel_size $EXPERT_PARALLEL_SIZE \
--expert_tensor_parallel_size $EXPERT_TENSOR_PARALLEL_SIZE \
--recompute_modules $RECOMPUTE_MODULES \
--cuda_graph_implementation $CUDA_GRAPH_IMPLEMENTATION \
--cuda_graph_scope $CUDA_GRAPH_SCOPE \
--lr $LR \
--eval_check_interval $EVAL_CHECK_INTERVAL \
--eval_batches $EVAL_BATCHES \
$CMD_SUFFIX
