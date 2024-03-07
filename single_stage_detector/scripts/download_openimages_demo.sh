#!/bin/bash

: "${DATASET_PATH:=/datasets/open-images-v6-mlperf}"

while [ "$1" != "" ]; do
  case $1 in
  -d | --dataset-path)
    shift
    DATASET_PATH=$1
    ;;
  --data_dir=*)
    if [[ "$PWD" = /workspace/single_stage_detector/ssd ]]; then
      cd ../scripts
      DATASET_PATH="${1#*=}"
    fi
    ;;
  esac
  shift
done

echo "saving to"
echo $DATASET_PATH
ls $DATASET_PATH

MLPERF_CLASSES=('Apple' 'Banana')

python fiftyone_openimages.py \
  --dataset-dir=${DATASET_PATH} \
  --output-labels="openimages-mlperf.json" \
  --classes "${MLPERF_CLASSES[@]}"
