#!/bin/bash

set +x
set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Set variables
: "${TFDATA_PATH:=./workspace/output_data}"
: "${INIT_CHECKPOINT:=./workspace/data/tf2_ckpt}"
: "${EVAL_FILE:=./workspace/tf_eval_data/eval_10k}"
: "${CONFIG_PATH:=./workspace/data/bert_config.json}"
: "${LOG_DIR:=./workspace/logs}"
: "${OUTPUT_DIR:=./workspace/final_output}"

# Handle MLCube parameters
while [ $# -gt 0 ]; do
  case "$1" in
  --tfdata_path=*)
    TFDATA_PATH="${1#*=}"
    ;;
  --config_path=*)
    CONFIG_PATH="${1#*=}"
    ;;
  --init_checkpoint=*)
    INIT_CHECKPOINT="${1#*=}"
    ;;
  --log_dir=*)
    LOG_DIR="${1#*=}"
    ;;
  --output_dir=*)
    OUTPUT_DIR="${1#*=}"
    ;;
  --eval_file=*)
    EVAL_FILE="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

# run benchmark
echo "running benchmark"

TF_XLA_FLAGS='--tf_xla_auto_jit=2' \
  python run_pretraining.py \
  --bert_config_file=$CONFIG_PATH \
  --output_dir=$OUTPUT_DIR \
  --input_file="$TFDATA_PATH/part*" \
  --init_checkpoint=$INIT_CHECKPOINT \
  --nodo_eval \
  --do_train \
  --eval_batch_size=4 \
  --learning_rate=0.0001 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=107538 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=6250 \
  --start_warmup_step=0 \
  --num_gpus=1 \
  --train_batch_size=12 |& tee "$LOG_DIR/train_console.log"

# Copy log file to MLCube log folder
if [ "$LOG_DIR" != "" ]; then
  timestamp=$(date +%Y%m%d_%H%M%S)
  cp bert.log "$LOG_DIR/bert_train_$timestamp.log"
fi

TF_XLA_FLAGS='--tf_xla_auto_jit=2' \
  python3 run_pretraining.py \
  --bert_config_file=$CONFIG_PATH \
  --output_dir=$OUTPUT_DIR \
  --input_file=$EVAL_FILE \
  --do_eval \
  --nodo_train \
  --eval_batch_size=8 \
  --init_checkpoint=$OUTPUT_DIR/model.ckpt-107538 \
  --iterations_per_loop=1000 \
  --learning_rate=0.0001 \
  --max_eval_steps=1250 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_gpus=1 \
  --num_train_steps=107538 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=1562 \
  --start_warmup_step=0 \
  --train_batch_size=24 \
  --nouse_tpu |& tee "$LOG_DIR/eval_console.log"

# Copy log file to MLCube log folder
if [ "$LOG_DIR" != "" ]; then
  timestamp=$(date +%Y%m%d_%H%M%S)
  cp bert.log "$LOG_DIR/bert_eval_$timestamp.log"
fi

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
