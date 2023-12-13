#!/bin/bash
set -e

: "${data_dir:=/}"

while [ $# -gt 0 ]; do
  case "$1" in
  --data_dir=*)
    DATA_DIR="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

python preprocess_dataset.py --data_dir $DATA_DIR --results_dir $DATA_DIR