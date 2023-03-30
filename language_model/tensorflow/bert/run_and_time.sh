#!/bin/bash

set +x
set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Set variables
: "${TFDATA_PATH:=./workspace/output_data}"
: "${CONFIG_PATH:=./workspace/data/bert_config.json}"
: "${LOG_DIR:=./workspace/logs}"

# Handle MLCube parameters
while [ $# -gt 0 ]; do
  case "$1" in
  --tfdata_path=*)
    TFDATA_PATH="${1#*=}"
    ;;
  --config_path=*)
    CONFIG_PATH="${1#*=}"
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
  --output_dir=/tmp/output/ \
  --input_file=$TFDATA_PATH \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --learning_rate=0.0001 \
  --init_checkpoint=./checkpoint/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=107538 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=6250 \
  --start_warmup_step=0 \
  --num_gpus=8 \
  --train_batch_size=24

# Copy log file to MLCube log folder
if [ "$LOG_DIR" != "" ]; then
  timestamp=$(date +%Y%m%d_%H%M%S)
  cp mlperf_compliance.log "$LOG_DIR/mlperf_compliance_$timestamp.log"
fi

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
