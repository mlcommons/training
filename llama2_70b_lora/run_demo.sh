#!/bin/bash

set +x
set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Set variables
: "${DATA_DIR:=./dataset}"
: "${MODEL_DIR:=./models/Llama-2-7b-chat-hf}"
: "${RESULT_DIR:=./workspace/results}"
: "${CONFIG_PATH:=./configs/default_config.yaml}"
: "${LOG_DIR:=./workspace/logs}"

# Handle MLCube parameters
while [ $# -gt 0 ]; do
    case "$1" in
    --data_dir=*)
        DATA_DIR="${1#*=}"
        ;;
    --model_dir=*)
        MODEL_DIR="${1#*=}"
        ;;
    --result_dir=*)
        RESULT_DIR="${1#*=}"
        ;;
    --config_path=*)
        CONFIG_PATH="${1#*=}"
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

accelerate launch --config_file $CONFIG_PATH scripts/train.py \
    --dataset_path $DATA_DIR/scrolls_gov_report_8k \
    --model_path $MODEL_DIR/Llama-2-7b-chat-hf \
    --max_seq_len 8192 \
    --bf16 True \
    --logging_steps 1 \
    --eval_steps 1 \
    --output_dir $RESULT_DIR/llama-70b_scrolls_gov_report_r16_$1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type "cosine" \
    --learning_rate 4e-4 \
    --weight_decay 0.0001 \
    --warmup_ratio 0 \
    --max_grad_norm 0.3 \
    --use_gradient_checkpointing True \
    --target_eval_loss 0.925 \
    --use_peft_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --max_steps 2 \
    --lora_target_modules "qkv_proj,o_proj" |& tee "$LOG_DIR/train_console.log"

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
