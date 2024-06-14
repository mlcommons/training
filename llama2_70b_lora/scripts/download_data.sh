#!/bin/bash

DATA_DIR="./data"
MODEL_DIR="./model"

# Capture MLCube parameter
while [ $# -gt 0 ]; do
    case "$1" in
    --data_dir=*)
        DATA_DIR="${1#*=}"
        ;;
    --model_dir=*)
        MODEL_DIR="${1#*=}"
        ;;
    *) ;;
    esac
    shift
done

mkdir -p $DATA_DIR

cd $DATA_DIR

rclone copy mlc-llama2:Llama2-70b-fused-qkv-mlperf ./Llama2-70b-fused-qkv-mlperf -P

cd -

mkdir -p $MODEL_DIR

cd $MODEL_DIR

rclone copy mlc-llama2:training/scrolls_gov_report_8k ./scrolls_gov_report_8k -P
