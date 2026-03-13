#!/bin/bash
set -e

DATA_PATH=""
MODEL_PATH=""
LOG_DIR=""

# Capture MLCube parameter
while [ $# -gt 0 ]; do
  case "$1" in
  --data_path=*)
    DATA_PATH="${1#*=}"
    ;;
  --model_path=*)
    MODEL_PATH="${1#*=}"
    ;;
  --log_dir=*)
    LOG_DIR="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

if [[ -z "$DATA_PATH" || -z "$MODEL_PATH" || -z "$LOG_DIR" ]]; then
  echo "Error: --data_path and --log_dir were not provided by MLCube." >&2
  exit 1
fi

echo "Data Path: $DATA_PATH"
echo "Model Path: $MODEL_PATH"
echo "Log Directory: $LOG_DIR"
echo "--------------------------"

export DATAROOT="$DATA_PATH"
export MODELROOT="$MODEL_PATH"
export LOGDIR="$LOG_DIR"
export NGPU=1
export CONFIG_FILE="torchtitan/torchtitan/experiments/flux/train_configs/flux_schnell_mlperf_preprocessed.toml"

echo "Running training with the following environment:"
echo "DATAROOT=$DATAROOT"
echo "MODELROOT=$MODELROOT"
echo "LOGDIR=$LOGDIR"
echo "NGPU=$NGPU"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "--------------------------"

ln -s $DATAROOT /dataset
ln -s $MODELROOT /models

bash torchtitan/torchtitan/experiments/flux/run_train.sh \
  --training.steps=10 \
  --training.batch_size=1 \
  --training.seq_len=2 \
  --eval.eval_freq=5
