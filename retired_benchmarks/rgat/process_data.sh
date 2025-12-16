#!/bin/bash

DATA_DIR="./igbh/full/processed"

# Capture MLCube parameter
while [ $# -gt 0 ]; do
    case "$1" in
    --data_dir=*)
        DATA_DIR="${1#*=}"
        ;;
    *) ;;
    esac
    shift
done

echo "Dataset processing starting"
python split_seeds.py --dataset_size='full' --path $DATA_DIR
echo "Dataset processing finished"