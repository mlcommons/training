#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

echo -e "\nNVIDIA container build: ${NVIDIA_BUILD_ID}\n"

export OMP_NUM_THREADS=1

DATA_DIR=${DATA_DIR:-"/datasets/LibriSpeech"}
MODEL_CONFIG=${MODEL_CONFIG:-"configs/rnnt.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"/results"}
CHECKPOINT=${CHECKPOINT:-""}
CREATE_LOGFILE=${CREATE_LOGFILE:-"true"}
CUDNN_BENCHMARK=${CUDNN_BENCHMARK:-"true"}
NUM_GPUS=${NUM_GPUS:-8}
AMP=${AMP:-"true"}
EPOCHS=${EPOCHS:-100}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-6}  # 8000 steps with 1x8x24 should be ~5.6 epochs
HOLD_EPOCHS=${HOLD_EPOCHS:-0}
SEED=${SEED:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-2}
OPTIMIZER=${OPTIMIZER:-"adamw"}
LEARNING_RATE=${LEARNING_RATE:-"0.001"}
LR_POLICY=${LR_POLICY:-"legacy"}
# LR_EXP_GAMMA=${LR_EXP_GAMMA:-0.981}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
EMA=${EMA:-0.0}  # XXX
SAVE_FREQUENCY=${SAVE_FREQUENCY:-10}
EPOCHS_THIS_JOB=${EPOCHS_THIS_JOB:-0}
RESUME=${RESUME:-"true"}
DALI_DEVICE=${DALI_DEVICE:-"none"}

mkdir -p "$OUTPUT_DIR"

ARGS=" --batch_size=$BATCH_SIZE"
ARGS+=" --val_batch_size=$VAL_BATCH_SIZE"
ARGS+=" --output_dir=$OUTPUT_DIR"
ARGS+=" --model_config=$MODEL_CONFIG"
ARGS+=" --lr=$LEARNING_RATE"
ARGS+=" --min_lr=1e-5"
ARGS+=" --lr_policy=$LR_POLICY"
# ARGS+=" --lr_exp_gamma=$LR_EXP_GAMMA"
ARGS+=" --epochs=$EPOCHS"
ARGS+=" --warmup_epochs=$WARMUP_EPOCHS"
ARGS+=" --hold_epochs=$HOLD_EPOCHS"
ARGS+=" --epochs_this_job=$EPOCHS_THIS_JOB"
ARGS+=" --ema=$EMA"
ARGS+=" --seed=$SEED"
ARGS+=" --optimizer=$OPTIMIZER"
ARGS+=" --dataset_dir=$DATA_DIR"
ARGS+=" --val_manifest=$DATA_DIR/librispeech-dev-clean-wav.json"
ARGS+=" --train_manifest=$DATA_DIR/librispeech-bench-clean-wav.json"
# ARGS+=",$DATA_DIR/librispeech-train-clean-360-wav.json"
# ARGS+=",$DATA_DIR/librispeech-train-other-500-wav.json"
ARGS+=" --weight_decay=1e-3"
ARGS+=" --save_frequency=$SAVE_FREQUENCY"
ARGS+=" --eval_frequency=1000"  # XXX =100
ARGS+=" --train_frequency=1"
ARGS+=" --print_prediction_frequency=100"
ARGS+=" --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS "
ARGS+=" --dali_device=$DALI_DEVICE"
[ "$AMP" == "true" ] && \
ARGS+=" --amp"
[ "$RESUME" == "true" ] && \
ARGS+=" --resume"
[ "$CUDNN_BENCHMARK" = "true" ] && \
ARGS+=" --cudnn_benchmark"
[ -n "$CHECKPOINT" ] && \
ARGS+=" --ckpt=${CHECKPOINT}"

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train.py $ARGS
