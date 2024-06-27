#!/bin/bash
set -e

: "${DATASET_PATH:=/}"

while [ "$1" != "" ]; do
    case $1 in
    --data_dir=*)
        DATASET_PATH="${1#*=}"
        ;;
    esac
    shift
done

wget https://storage.googleapis.com/mlperf_training_demo/3d_unet/demo_data.zip
unzip -o demo_data.zip -d $DATASET_PATH