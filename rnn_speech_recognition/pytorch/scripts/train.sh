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

export OMP_NUM_THREADS=1

: ${DATA_DIR:=${1:-"/datasets/LibriSpeech"}}
: ${MODEL_CONFIG:=${2:-"configs/baseline_v3-1023sp.yaml"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${CHECKPOINT:=${4:-}}
: ${CUDNN_BENCHMARK:=true}
: ${NUM_GPUS:=8}
: ${AMP:=false}
: ${BATCH_SIZE:=8}
: ${VAL_BATCH_SIZE:=8}
: ${OPTIMIZER:=adamw}
: ${GRAD_ACCUMULATION_STEPS:=1}
: ${LEARNING_RATE:=0.001}
# : ${MIN_LEARNING_RATE:=0.00001}
: ${LR_POLICY:=legacy}
: ${LR_EXP_GAMMA:=0.935}  # ~0.005 in 80 epochs
: ${NUM_BUCKETS:=6} # empty means to use torch.utils.data.distributed.DistributedSampler
: ${EMA:=0.999}
: ${SEED:=1}
: ${EPOCHS:=100}
: ${WARMUP_EPOCHS:=6}  # 8000 steps with 1x8x24 should be ~5.6 epochs
: ${HOLD_EPOCHS:=0}
: ${SAVE_FREQUENCY:=10}
: ${EPOCHS_THIS_JOB:=0}
: ${RESUME:=true}
: ${DALI_DEVICE:="cpu"}
: ${PAD_TO_MAX_DURATION:=false}
: ${VAL_FREQUENCY:=1}
: ${PREDICTION_FREQUENCY:=1000}
: ${B1:=0.9}
: ${B2:=0.999}
: ${LOG_FREQUENCY:=1}
: ${TRAIN_MANIFESTS:="$DATA_DIR/librispeech-train-clean-100-wav.json \
                      $DATA_DIR/librispeech-train-clean-360-wav.json \
                      $DATA_DIR/librispeech-train-other-500-wav.json"}
: ${VAL_MANIFESTS:="$DATA_DIR/librispeech-dev-clean-wav.json"}
: ${SAVE:=true}

mkdir -p "$OUTPUT_DIR"

ARGS="--dataset_dir=$DATA_DIR"
ARGS+=" --val_manifests $VAL_MANIFESTS"
ARGS+=" --train_manifests $TRAIN_MANIFESTS"
ARGS+=" --model_config=$MODEL_CONFIG"
ARGS+=" --output_dir=$OUTPUT_DIR"
ARGS+=" --lr=$LEARNING_RATE"
ARGS+=" --batch_size=$BATCH_SIZE"
ARGS+=" --val_batch_size=$VAL_BATCH_SIZE"
ARGS+=" --min_lr=1e-5"
ARGS+=" --lr_policy=$LR_POLICY"
ARGS+=" --lr_exp_gamma=$LR_EXP_GAMMA"
ARGS+=" --epochs=$EPOCHS"
ARGS+=" --warmup_epochs=$WARMUP_EPOCHS"
ARGS+=" --hold_epochs=$HOLD_EPOCHS"
ARGS+=" --epochs_this_job=$EPOCHS_THIS_JOB"
ARGS+=" --ema=$EMA"
ARGS+=" --seed=$SEED"
ARGS+=" --optimizer=$OPTIMIZER"
ARGS+=" --weight_decay=1e-3"
ARGS+=" --save_frequency=$SAVE_FREQUENCY"
ARGS+=" --log_frequency=$LOG_FREQUENCY"
ARGS+=" --val_frequency=$VAL_FREQUENCY"
ARGS+=" --prediction_frequency=$PREDICTION_FREQUENCY"
ARGS+=" --grad_accumulation_steps=$GRAD_ACCUMULATION_STEPS "
ARGS+=" --dali_device=$DALI_DEVICE"
ARGS+=" --b1=$B1"
ARGS+=" --b2=$B2"

[ "$AMP" = true ] &&                 ARGS+=" --amp"
[ "$RESUME" = true ] &&              ARGS+=" --resume"
[ "$CUDNN_BENCHMARK" = true ] &&     ARGS+=" --cudnn_benchmark"
[ "$SAVE" = true ] &&                ARGS+=" --keep_milestones 50 100 150 200"
[ "$SAVE" = true ] &&                ARGS+=" --save_best_from=80"
[ -n "$CHECKPOINT" ] &&              ARGS+=" --ckpt=$CHECKPOINT"
[ -n "$NUM_BUCKETS" ] &&             ARGS+=" --num_buckets=$NUM_BUCKETS"
[ -n "$TARGET" ] &&                  ARGS+=" --target=$TARGET"
[ -n "$CLIP_NORM" ] &&               ARGS+=" --clip_norm=$CLIP_NORM"
[ -n "$START_CLIP" ] &&              ARGS+=" --start_clip=$START_CLIP"

DISTRIBUTED=${DISTRIBUTED:-"-m torch.distributed.launch --nproc_per_node=$NUM_GPUS"}
python ${DISTRIBUTED} train.py ${ARGS}
