#!/bin/bash

: "${NUM_NODES:=1}"
: "${GPUS_PER_NODE:=8}"
: "${CHECKPOINT:=/checkpoints/sd/512-base-ema.ckpt}"
: "${RESULTS_DIR:=}"
: "${CONFIG:=./configs/train_512_latents.yaml}"

while [ "$1" != "" ]; do
    case $1 in
        --num-nodes )           shift
                                NUM_NODES=$1
                                ;;
        --gpus-per-node )       shift
                                GPUS_PER_NODE=$1
                                ;;
        --checkpoint )          shift
                                CHECKPOINT=$1
                                ;;
        --results-dir )         shift
                                RESULTS_DIR=$1
                                ;;
        --config )              shift
                                CONFIG=$1
                                ;;
    esac
    shift
done

set -e

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HOME=/hf_home

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# CLEAR YOUR CACHE HERE
python -c "
from mlperf_logging.mllog import constants
from mlperf_logging_utils import mllogger
mllogger.event(key=constants.CACHE_CLEAR, value=True)"

python main.py \
    lightning.trainer.num_nodes=${NUM_NODES} \
    lightning.trainer.devices=${GPUS_PER_NODE} \
    -m train \
    --ckpt ${CHECKPOINT} \
    --logdir ${RESULTS_DIR}  \
    -b ${CONFIG}

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# runtime
runtime=$(( $end - $start ))
result_name="stable_diffusion"

echo "RESULT,$result_name,$runtime,$USER,$start_fmt"
