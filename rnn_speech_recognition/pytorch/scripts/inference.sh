#!/bin/bash

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

# TODO Check if NUM_STEPS is still supported and how is set in infer/bench
# TODO Check if multi-gpu works ok

: ${DATA_DIR:=${1:-"/datasets/LibriSpeech"}}
: ${MODEL_CONFIG:=${2:-"configs/rnnt.yaml"}}
: ${OUTPUT_DIR:=${3:-"/results"}}
: ${CHECKPOINT:=${4:-"/checkpoints/rnnt_fp16.pt"}}
: ${DATASET:="dev-clean"}
: ${CUDNN_BENCHMARK:=false}
: ${MAX_DURATION:=""}
: ${PAD_TO_MAX_DURATION:=false}
: ${NUM_GPUS:=1}
: ${NUM_STEPS:="-1"}
: ${AMP:=true}
: ${BATCH_SIZE:=8}
: ${EMA:=false}
: ${SEED:=0}
: ${DALI_DEVICE:="none"}
: ${CPU:=false}
: ${LOGITS_FILE:=}
: ${PREDICTION_FILE="${OUTPUT_DIR}/${DATASET}.predictions"}
: ${REPEATS:=1}

mkdir -p "$OUTPUT_DIR"

ARGS="--dataset_dir=$DATA_DIR"
ARGS+=" --val_manifest=$DATA_DIR/librispeech-${DATASET}-wav.json"
ARGS+=" --model_config=$MODEL_CONFIG"
ARGS+=" --output_dir=$OUTPUT_DIR"
ARGS+=" --batch_size=$BATCH_SIZE"
ARGS+=" --seed=$SEED"
ARGS+=" --dali_device=$DALI_DEVICE"
ARGS+=" --repeats=$REPEATS"

[ "$AMP" = true ] &&                 ARGS+=" --amp"
[ "$EMA" = true ] &&                 ARGS+=" --ema"
[ "$CUDNN_BENCHMARK" = true ] &&     ARGS+=" --cudnn_benchmark"
[ -n "$CHECKPOINT" ] &&              ARGS+=" --ckpt=$CHECKPOINT"
[ "$NUM_STEPS" -gt 0 ] &&            ARGS+=" --steps $NUM_STEPS"
[ -n "$PREDICTION_FILE" ] &&         ARGS+=" --save_prediction $PREDICTION_FILE"
[ -n "$LOGITS_FILE" ] &&             ARGS+=" --logits_save_to $LOGITS_FILE"
[ "$CPU" = true ] &&                 ARGS+=" --cpu"
[ -n "$MAX_DURATION" ] &&            ARGS+=" --max_duration $MAX_DURATION"
[ "$PAD_TO_MAX_DURATION" = true ] && ARGS+=" --pad_to_max_duration"

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS inference.py $ARGS
