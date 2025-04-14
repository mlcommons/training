#!/bin/bash

DATA_DIR="./data"
MODEL_DIR="./model"
RCLONE_CONFIG="./rclone.conf"

# Capture MLCube parameter
while [ $# -gt 0 ]; do
    case "$1" in
    --data_dir=*)
        DATA_DIR="${1#*=}"
        ;;
    --model_dir=*)
        MODEL_DIR="${1#*=}"
        ;;
    --rclone_config=*)
        RCLONE_CONFIG="${1#*=}"
        ;;
    *) ;;
    esac
    shift
done

mkdir -p ~/.config/rclone/
cp $RCLONE_CONFIG ~/.config/rclone/rclone.conf

mkdir -p $DATA_DIR

cd $DATA_DIR

rclone copy mlc-training:mlcommons-training-wg-public/common/datasets/c4/mixtral_8x22b_preprocessed $DATA_DIR/mixtral_8x22b_preprocessed -P
rclone copy mlc-training:mlcommons-training-wg-public/llama3_1/datasets/tokenizer $DATA_DIR/tokenizer -P

cd -

mkdir -p $MODEL_DIR

cd $MODEL_DIR

rclone copy mlc-llama3-1:training/nemo-formatted-hf-checkpoint/8b ./nemo-formatted-hf-checkpoint/8b -P