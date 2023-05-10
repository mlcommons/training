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



python3 ./run_pretraining.py \
	--bert_config_file=gs://bert_tf_data/bert_config.json \
	--nodo_eval \
	--do_train \
	--eval_batch_size=640 \
	--init_checkpoint=gs://bert_tf_data/tf2_ckpt/model.ckpt-28252 \
	--input_file=gs://bert_tf_data/tf_data/part-* \
	--iterations_per_loop=3 \
	--lamb_beta_1=0.88 \
	--lamb_beta_2=0.88 \
	--lamb_weight_decay_rate=0.0166629 \
	--learning_rate=0.00288293 \
	--log_epsilon=-6 \
	--max_eval_steps=125 \
	--max_predictions_per_seq=76 \
	--max_seq_length=512 \
	--num_tpu_cores=128 \
	--num_train_steps=600 \
	--num_warmup_steps=287 \
	--optimizer=lamb \
	--output_dir=gs://bert_tf_data/output/ \
	--save_checkpoints_steps=3 \
	--start_warmup_step=-76 \
	--steps_per_update=1 \
	--train_batch_size=8192 \
	--use_tpu \
	--tpu_name=tpu_test \
	--tpu_zone=europe-west4-a \
	--gcp_project=training-reference-bench-test |& tee "$LOG_DIR/train_console.log"



set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
