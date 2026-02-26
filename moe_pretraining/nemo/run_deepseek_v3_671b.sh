#!/bin/bash

# Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
#     Slurm settings
: "${ACCOUNT:?ACCOUNT not set}"
: "${PARTITION:?PARTITION not set}"

#     Job settings
: "${LOG_DIR:?LOG_DIR not set}"
: "${IMAGE:?IMAGE not set}"

#     Dataset settings
: "${DATA_DIR:?DATA_DIR not set}"
: "${MODEL_CKPT:?MODEL_CKPT not set}"

# Vars with defaults
#     Slurm settings
: "${TIME:="04:00:00"}"
: "${NNODES:=64}"
: "${GPUS_PER_NODE:=4}"
: "${GPU:="gb300"}"
: "${SEGMENT:=""}"


#     Model settings
: "${GBS:=2048}"
: "${MBS:=1}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${PIPELINE_PARALLEL_SIZE:=4}"
: "${VIRTUAL_PIPELINE_PARALLEL_SIZE:=""}"
: "${CONTEXT_PARALLEL_SIZE:=1}"
: "${EXPERT_PARALLEL_SIZE:=64}"
: "${EXPERT_TENSOR_PARALLEL_SIZE:=1}"
: "${SEQUENCE_LENGTH:=4096}"
: "${RECOMPUTE_MODULES:="mlp,moe_act"}"
: "${MOE_TOKEN_DISPATCHER_TYPE:="alltoall"}"
: "${MOE_GROUPED_GEMM:=True}"
: "${MOE_PERMUTE_FUSION:=False}"
: "${MOE_ROUTER_FUSION:=False}"

#     Training settings
: "${MAX_STEPS:=1000}"
: "${WARMUP_STEPS:=0}"
: "${MAX_LR:="2e-4"}"
: "${MIN_LR:="5e-6"}"
: "${SEED:=1234}"

#     Eval settings
: "${EVAL_CHECK_INTERVAL:=10}"
: "${EVAL_BATCHES:=1}"
: "${EVAL_BATCH_SIZE:=""}"

#     Experiment settings
: "${EXP_NAME:=""}"
: "${TARGET:="3.60"}"
: "${DRYRUN:=0}"
: "${DETACH:=1}"

# Build mounts
MOUNTS="${LOG_DIR}:/output,${LOG_DIR}:/mlperf-outputs,${DATA_DIR}:/preproc_data,${MODEL_CKPT}:/checkpoint,${DATA_DIR}/tokenizer:/tokenizer"

TMP_NPY_INDEX="$LOG_DIR/npy_index"
mkdir -p "$TMP_NPY_INDEX"
MOUNTS="${MOUNTS},${TMP_NPY_INDEX}:/npy_index"

# Build launcher arguments
LAUNCHER_ARGS="--account $ACCOUNT --partition $PARTITION"
LAUNCHER_ARGS="$LAUNCHER_ARGS --nodes $NNODES --gpus_per_node $GPUS_PER_NODE"
LAUNCHER_ARGS="$LAUNCHER_ARGS --gpu $GPU"
LAUNCHER_ARGS="$LAUNCHER_ARGS --time_limit $TIME"
if [ -n "$SEGMENT" ]; then
    LAUNCHER_ARGS="$LAUNCHER_ARGS --segment $SEGMENT"
fi
LAUNCHER_ARGS="$LAUNCHER_ARGS --container_image $IMAGE"
LAUNCHER_ARGS="$LAUNCHER_ARGS --log_dir $LOG_DIR"
LAUNCHER_ARGS="$LAUNCHER_ARGS --mounts $MOUNTS"

if [ -n "$EXP_NAME" ]; then
    LAUNCHER_ARGS="$LAUNCHER_ARGS --exp_name $EXP_NAME"
fi

if [ "$DRYRUN" -gt 0 ]; then
    LAUNCHER_ARGS="$LAUNCHER_ARGS --dryrun"
fi

if [ "$DETACH" -gt 0 ]; then
    LAUNCHER_ARGS="$LAUNCHER_ARGS --detach"
fi

# Build pretrain arguments
PRETRAIN_ARGS="--nodes $NNODES --gpus_per_node $GPUS_PER_NODE"
PRETRAIN_ARGS="$PRETRAIN_ARGS --tensor_parallel_size $TENSOR_PARALLEL_SIZE"
PRETRAIN_ARGS="$PRETRAIN_ARGS --pipeline_parallel_size $PIPELINE_PARALLEL_SIZE"
PRETRAIN_ARGS="$PRETRAIN_ARGS --context_parallel_size $CONTEXT_PARALLEL_SIZE"
PRETRAIN_ARGS="$PRETRAIN_ARGS --expert_model_parallel_size $EXPERT_PARALLEL_SIZE"
PRETRAIN_ARGS="$PRETRAIN_ARGS --expert_tensor_parallel_size $EXPERT_TENSOR_PARALLEL_SIZE"
PRETRAIN_ARGS="$PRETRAIN_ARGS --sequence_length $SEQUENCE_LENGTH"
PRETRAIN_ARGS="$PRETRAIN_ARGS --gbs $GBS"
PRETRAIN_ARGS="$PRETRAIN_ARGS --mbs $MBS"
PRETRAIN_ARGS="$PRETRAIN_ARGS --lr $MAX_LR"
PRETRAIN_ARGS="$PRETRAIN_ARGS --min_lr $MIN_LR"
PRETRAIN_ARGS="$PRETRAIN_ARGS --max_steps $MAX_STEPS"
PRETRAIN_ARGS="$PRETRAIN_ARGS --warmup_steps $WARMUP_STEPS"
PRETRAIN_ARGS="$PRETRAIN_ARGS --seed $SEED"
PRETRAIN_ARGS="$PRETRAIN_ARGS --eval_check_interval $EVAL_CHECK_INTERVAL"
PRETRAIN_ARGS="$PRETRAIN_ARGS --eval_batches $EVAL_BATCHES"
if [ -n "$EVAL_BATCH_SIZE" ]; then
    PRETRAIN_ARGS="$PRETRAIN_ARGS --eval_batch_size $EVAL_BATCH_SIZE"
fi
PRETRAIN_ARGS="$PRETRAIN_ARGS --target_log_ppl $TARGET"

if [ -n "$VIRTUAL_PIPELINE_PARALLEL_SIZE" ]; then
    PRETRAIN_ARGS="$PRETRAIN_ARGS --virtual_pipeline_parallel_size $VIRTUAL_PIPELINE_PARALLEL_SIZE"
fi

if [ -n "$RECOMPUTE_MODULES" ]; then
    PRETRAIN_ARGS="$PRETRAIN_ARGS --recompute_modules $RECOMPUTE_MODULES"
fi


PRETRAIN_ARGS="$PRETRAIN_ARGS --moe_token_dispatcher_type $MOE_TOKEN_DISPATCHER_TYPE"
PRETRAIN_ARGS="$PRETRAIN_ARGS --moe_grouped_gemm $MOE_GROUPED_GEMM"
PRETRAIN_ARGS="$PRETRAIN_ARGS --moe_permute_fusion $MOE_PERMUTE_FUSION"
PRETRAIN_ARGS="$PRETRAIN_ARGS --moe_router_fusion $MOE_ROUTER_FUSION"

# Allows MLLogger objects to be constructed locally
if [ ! -d /mlperf-outputs ]; then mkdir -p /mlperf-outputs 2>/dev/null || true; fi

set -x

python3 run_deepseek.py \
    $LAUNCHER_ARGS \
    -- \
    $PRETRAIN_ARGS
