#!/bin/bash
set -e

DATA_DIR="${DATA_DIR:-/raw-data-dir/kits19/data}"
PROCESSED_DIR="${PROCESSED_DIR:-/raw-data-dir/processed_data}"
if [ -z "$(ls -A "$PROCESSED_DIR")" ]
then
    ARGS="--data_dir=$DATA_DIR"
    ARGS+=" --results_dir $PROCESSED_DIR"
    echo "Starting dataset preprocessing - This may take a while..."
    python3 preprocess_dataset.py ${ARGS}
else
    echo "Directory $PROCESSED_DIR is not empty."
fi
