#!/bin/bash

export DGXSYSTEM=B200_1x8x1
export GPUS_PER_NODE=8
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29501

export PRIMUS_PATH=/workspace/deps/Primus
export PYTHONPATH="${PRIMUS_PATH}:${PRIMUS_PATH}/third_party/Megatron-LM:${PYTHONPATH}"
export EXP=/workspace/code/conf/gpt_oss_20B-pretrain-nvidia.yaml
export DATA_PATH=/data
export MODEL=/model

export PRIMUS_MICRO_BATCH_SIZE=2
export PRIMUS_GLOBAL_BATCH_SIZE=16
export PRIMUS_LR=4.0e-4
export PRIMUS_MIN_LR=4.0e-5             # Set to 10% of max LR
export PRIMUS_TRAIN_ITERS=1200000       # 1.2M iters × 16 GBS = 19.2B samples
export PRIMUS_LR_WARMUP_ITERS=128
export PRIMUS_LR_DECAY_ITERS=$((PRIMUS_TRAIN_ITERS-PRIMUS_LR_WARMUP_ITERS))

# Evaluation frequency (sample-based, adjusts automatically with GBS)
export EVAL_SAMPLES_INTERVAL=12288   # Evaluate every 12,288 samples
export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))  # Auto-computed

export PRIMUS_BF16=true
export PRIMUS_FP16=false
export PRIMUS_FP8=null

export PRIMUS_TURBO_ENABLED=false
export USE_TURBO_ATTENTION=false
export USE_TURBO_GROUPED_MLP=false
export USE_TURBO_DEEPEP=false
export TURBO_DEEPEP_NUM_CU=0
export TURBO_SYNC_FREE_MOE_STAGE=0

export PRIMUS_APPLY_ROPE_FUSION=false
export USE_ROCM_MEM_INFO=false

export OVERLAP_GRAD_REDUCE=true
export OVERLAP_PARAM_GATHER=true
export GRADIENT_ACCUMULATION_FUSION=false

export PRIMUS_TP=1
export PRIMUS_PP=1
export PRIMUS_EP=8

export ENABLE_MLLOG=1
export MLLOG_OUTPUT_FILE=/results/mlperf_output.log
export MLLOG_TRAIN_LOSS_LOG_FREQ=32
export MLLOG_TARGET_EVAL_LOSS=3.34
export MLLOG_SUBMISSION_BENCHMARK=gpt-oss-20b
export MLLOG_SUBMISSION_DIVISION=closed
export MLLOG_SUBMISSION_ORG=NVIDIA
export MLLOG_SUBMISSION_PLATFORM=B200

export HF_TOKEN="${HF_TOKEN:-}"

