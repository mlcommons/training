#!/bin/bash
set -e

: "${data_dir:=/}"
: "${processed_data:=/}"

while [ $# -gt 0 ]; do
  case "$1" in
  --data_dir=*)
    DATA_DIR="${1#*=}"
    ;;
  --processed_data=*)
    PROCESSED_DATA="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

python preprocess_dataset.py --data_dir $DATA_DIR --results_dir $PROCESSED_DATA