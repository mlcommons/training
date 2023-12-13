#!/bin/bash
set -e

: "${DATASET_PATH:=/}"

while [ $# -gt 0 ]; do
  case "$1" in
  --dataset_path=*)
    DATASET_PATH="${1#*=}"
    ;;
  --processed_data_dir=*)
    PROCESSED_DATA_DIR="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

python preprocess_dataset.py --data_dir $DATASET_PATH --results_dir $PROCESSED_DATA_DIR