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

rclone copy mlc-llama2:training/scrolls_gov_report_8k ./scrolls_gov_report_8k -P

cd -

mkdir -p $MODEL_DIR

cd $MODEL_DIR

rclone copy mlc-llama2:Llama-2-7b-chat-hf ./Llama-2-7b-chat-hf -P
