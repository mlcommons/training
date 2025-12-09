#!/bin/bash

set +x
set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Set variables
: "${DATA_DIR:=./igbh/full/processed}"
: "${LOG_DIR:=./workspace/logs}"

# Handle MLCube parameters
while [ $# -gt 0 ]; do
    case "$1" in
    --data_dir=*)
        DATA_DIR="${1#*=}"
        ;;
    --log_dir=*)
        LOG_DIR="${1#*=}"
        ;;
    *) ;;
    esac
    shift
done

# run benchmark
echo "running benchmark"

python compress_graph.py --path $DATA_DIR \
    --dataset_size='full' \
    --layout='CSC' \
    --use_fp16 |& tee "$LOG_DIR/train_console.log"

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"