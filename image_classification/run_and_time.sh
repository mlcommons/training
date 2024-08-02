#!/bin/bash

set +x
set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Set variables
DATASET_DIR=${DATASET_DIR:-"/"}
LOG_DIR=${LOG_DIR:-"/"}
OUTPUT_MODEL_DIR=${OUTPUT_MODEL_DIR:-"/"}

# Handle MLCube parameters
while [ $# -gt 0 ]; do
  case "$1" in
  --data_dir=*)
    DATASET_DIR="${1#*=}"
    ;;
  --log_dir=*)
    LOG_DIR="${1#*=}"
    ;;
  --output_model_dir=*)
    OUTPUT_MODEL_DIR="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

# Check if labels file exist
if [ ! -f "$DATASET_DIR/synset_labels.txt" ]; then
    wget -O $DATASET_DIR/synset_labels.txt \
    https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
fi

export COMPLIANCE_FILE=$(pwd)/mlperf_compliance.log

# run benchmark
echo "running benchmark"

# run training
python3 tensorflow2/resnet_ctl_imagenet_main.py \
  --base_learning_rate=8.5 \
  --batch_size=1024 \
  --data_dir=${DATASET_DIR} \
  --datasets_num_private_threads=32 \
  --dtype=fp32 \
  --device_warmup_steps=1 \
  --noenable_device_warmup \
  --enable_eager \
  --noenable_xla \
  --epochs_between_evals=4 \
  --noeval_dataset_cache \
  --eval_offset_epochs=2 \
  --eval_prefetch_batchs=192 \
  --label_smoothing=0.1 \
  --lars_epsilon=0 \
  --log_steps=125 \
  --lr_schedule=polynomial \
  --model_dir=${OUTPUT_MODEL_DIR} \
  --momentum=0.9 \
  --num_accumulation_steps=2 \
  --num_classes=1000 \
  --num_gpus=8 \
  --optimizer=LARS \
  --noreport_accuracy_metrics \
  --single_l2_loss_op \
  --noskip_eval \
  --steps_per_loop=1252 \
  --target_accuracy=0.759 \
  --notf_data_experimental_slack \
  --tf_gpu_thread_mode=gpu_private \
  --notrace_warmup \
  --train_epochs=41 \
  --notraining_dataset_cache \
  --training_prefetch_batchs=128 \
  --nouse_synthetic_data \
  --warmup_epochs=5 \
  --weight_decay=0.0002 |& tee "$LOG_DIR/train_console.log"

# Copy log file to MLCube log folder
if [[ "$LOG_DIR" != "" && -f "$COMPLIANCE_FILE" ]]; then
  timestamp=$(date +%Y%m%d_%H%M%S)
  cp "$COMPLIANCE_FILE" "$LOG_DIR/mlperf_compliance_$timestamp.log"
fi

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(($end - $start))
result_name="IMAGE_CLASSIFICATION"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
