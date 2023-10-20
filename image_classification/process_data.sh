#!/bin/bash

: "${DATASET_PATH:=./}"
: "${PROCESSED_DATA_DIR:=./tfdata}"

while [ "$1" != "" ]; do
    case $1 in
    --data_dir=*)
        DATASET_PATH="${1#*=}"
        ;;
    --processed_data_dir=*)
        PROCESSED_DATA_DIR="${1#*=}"
        ;;
    esac
    shift
done

if [ ! -f "$DATASET_PATH/synset_labels.txt" ]; then
    if [ -f "$DATASET_PATH/words.txt" ]; then
        cp "$DATASET_PATH/words.txt" "$DATASET_PATH/synset_labels.txt"
    else
        echo "Missing file: synset_labels.txt"
    fi
fi


wget https://raw.githubusercontent.com/tensorflow/tpu/master/tools/datasets/imagenet_to_gcs.py

python imagenet_to_gcs.py \
  --raw_data_dir=$DATASET_PATH \
  --local_scratch_dir=$PROCESSED_DATA_DIR \
  --nogcs_upload
