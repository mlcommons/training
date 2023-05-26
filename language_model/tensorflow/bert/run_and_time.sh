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
: "${PARAMETERS_YAML:=./workspace/parameters.yaml}"

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

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

eval $(parse_yaml $PARAMETERS_YAML)

# run benchmark
echo "running benchmark"

python3 ./run_pretraining.py \
--bert_config_file=$CONFIG_PATH \
--nodo_eval \
--do_train \
--eval_batch_size=640 \
--init_checkpoint=$INIT_CHECKPOINT \
--input_file="$TFDATA_PATH/part*" \
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
--output_dir=$OUTPUT_DIR \
--save_checkpoints_steps=3 \
--start_warmup_step=-76 \
--steps_per_update=1 \
--train_batch_size=8192 \
--use_tpu \
--tpu_name=$tpu_name \
--tpu_zone=$tpu_zone \
--gcp_project=$gcp_project |& tee "$LOG_DIR/train_console.log"

# Copy log file to MLCube log folder
if [ "$LOG_DIR" != "" ]; then
  timestamp=$(date +%Y%m%d_%H%M%S)
  cp bert.log "$LOG_DIR/bert_train_$timestamp.log"
fi

python3 ./run_pretraining.py \
--bert_config_file=$CONFIG_PATH \
--do_eval \
--nodo_train \
--eval_batch_size=640 \
--init_checkpoint=$OUTPUT_DIR/model.ckpt-28252 \
--input_file=$EVAL_FILE \
--iterations_per_loop=3 \
--lamb_beta_1=0.88 \
--lamb_beta_2=0.88 \
--lamb_weight_decay_rate=0.0166629 \
--learning_rate=0.00288293 \
--log_epsilon=-6 \
--max_eval_steps=125 \
--max_predictions_per_seq=76 \
--max_seq_length=512 \
--num_tpu_cores=8 \
--num_train_steps=600 \
--num_warmup_steps=287 \
--optimizer=lamb \
--output_dir=$OUTPUT_DIR \
--save_checkpoints_steps=3 \
--start_warmup_step=-76 \
--steps_per_update=1 \
--train_batch_size=8192 \
--use_tpu \
--tpu_name=$tpu_name \
--tpu_zone=$tpu_zone \
--gcp_project=$gcp_project |& tee "$LOG_DIR/eval_console.log"

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
