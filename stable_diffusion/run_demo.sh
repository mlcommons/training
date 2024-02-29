#!/bin/bash

: "${NUM_NODES:=1}"
: "${GPUS_PER_NODE:=8}"
: "${CHECKPOINT_SD:=/checkpoints/sd/512-base-ema.ckpt}"
: "${CHECKPOINT_CLIP:=/checkpoints/clip/}"
: "${CHECKPOINT_INCEPTION:=/checkpoints/inception/}"
: "${COCO_DIR:=data_coco/}"
: "${LAION_DIR:=data_laion/}"
: "${RESULTS_DIR:=/results}"
: "${CONFIG:=./configs/train_demo.yaml}"


while [ $# -gt 0 ]; do
  case "$1" in
  --num-nodes=*)
    NUM_NODES="${1#*=}"
    ;;
  --gpus-per-node=*)
    GPUS_PER_NODE="${1#*=}"
    ;;
  --checkpoint_sd=*)
    CHECKPOINT_SD="${1#*=}"
    ;;
  --checkpoint_clip=*)
    CHECKPOINT_CLIP="${1#*=}"
    ;;
  --checkpoint_inception=*)
    CHECKPOINT_INCEPTION="${1#*=}"
    ;;
  --coco_dir=*)
    COCO_DIR="${1#*=}"
    ;;
  --laion_dir=*)
    LAION_DIR="${1#*=}"
    ;;
  --results_dir=*)
    RESULTS_DIR="${1#*=}"
    ;;
  --config=*)
    CONFIG="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

set -e

mkdir -p /checkpoints/clip
ln -s $CHECKPOINT_CLIP/* /checkpoints/clip

mkdir -p /datasets/coco2014
ln -s $COCO_DIR/* /datasets/coco2014

sed -i "s=/datasets/laion-400m/webdataset-moments-filtered/{00000..00831}.tar=$LAION_DIR/{00000..00003}.tar=g" $CONFIG
sed -i "s=/datasets/coco2014/val2014_512x512_30k_stats.npz=$COCO_DIR/val2014_30k_stats.npz=g" $CONFIG
sed -i "s=/results/inference=$RESULTS_DIR/=g" $CONFIG
sed -i "s=/checkpoints/clip=$CHECKPOINT_CLIP/=g" $CONFIG
sed -i "s=/checkpoints/inception=$CHECKPOINT_INCEPTION/=g" $CONFIG

export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export DIFFUSERS_OFFLINE=0
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
    --validation False \
    --ckpt ${CHECKPOINT_SD} \
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
