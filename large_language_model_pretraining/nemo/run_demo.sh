#!/bin/bash

set +x
set -e

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Set default variables (can be overridden via CLI)
: "${JOB_DIR:=./workspace/job}"
: "${PREPROCESSED_PATH:=./dataset/preprocessed}"
: "${TOKENIZER_PATH:=./tokenizer}"
: "${MODEL_CKPT:=./checkpoints}"
: "${CONTINUAL_CKPT:=./checkpoints/continual}"
: "${USE_CKPT:=0}"
: "${FROM_HF:=0}"
: "${SAVE_CKPT:=0}"
: "${MAX_RETRIES:=0}"
: "${SIZE:=405b}"
: "${GBS:=1}"
: "${MBS:=1}"
: "${START_STEPS:=0}"
: "${MAX_STEPS:=}"
: "${SEEDS:=42}"
: "${NEXP:=1}"
: "${NPAR:=1}"
: "${TAG:=default_run}"
: "${TARGET:=5.6}"
: "${STEP_TIME_ATOL:=7200}"

# Parse CLI args
while [ $# -gt 0 ]; do
    case "$1" in
        --job_dir=*) JOB_DIR="${1#*=}" ;;
        --preprocessed_path=*) PREPROCESSED_PATH="${1#*=}" ;;
        --tokenizer_path=*) TOKENIZER_PATH="${1#*=}" ;;
        --model_ckpt=*) MODEL_CKPT="${1#*=}" ;;
        --continual_ckpt=*) CONTINUAL_CKPT="${1#*=}" ;;
        --use_ckpt=*) USE_CKPT="${1#*=}" ;;
        --from_hf=*) FROM_HF="${1#*=}" ;;
        --save_ckpt=*) SAVE_CKPT="${1#*=}" ;;
        --max_retries=*) MAX_RETRIES="${1#*=}" ;;
        --size=*) SIZE="${1#*=}" ;;
        --gbs=*) GBS="${1#*=}" ;;
        --mbs=*) MBS="${1#*=}" ;;
        --start_steps=*) START_STEPS="${1#*=}" ;;
        --max_steps=*) MAX_STEPS="${1#*=}" ;;
        --seeds=*) SEEDS="${1#*=}" ;;
        --nexp=*) NEXP="${1#*=}" ;;
        --npar=*) NPAR="${1#*=}" ;;
        --tag=*) TAG="${1#*=}" ;;
        --target=*) TARGET="${1#*=}" ;;
        --step_time_atol=*) STEP_TIME_ATOL="${1#*=}" ;;
        *) ;;
    esac
    shift
done

CMD_SUFFIX=""
[[ "$USE_CKPT" -gt 0 ]] && CMD_SUFFIX+=" --use_ckpt"
[[ "$FROM_HF" -gt 0 ]] && CMD_SUFFIX+=" --resume_from_hf"
[[ "$SAVE_CKPT" -gt 0 ]] && CMD_SUFFIX+=" --save_ckpt"
[[ -n "$MAX_STEPS" ]] && CMD_SUFFIX+=" --max_steps $MAX_STEPS"
[[ -n "$TAG" ]] && CMD_SUFFIX+=" --tag $TAG"

IFS=" " read -ra seeds <<< "$SEEDS"

mkdir -p "$JOB_DIR/mlperf-outputs"

set -x

python3 pretrain_llama31.py \
    --job_dir "$JOB_DIR" \
    --nodes 1 \
    --gpus_per_node 1 \
    --size "$SIZE" \
    --gbs "$GBS" --mbs "$MBS" \
    --seeds "${seeds[@]}" \
    --num_exps "$NEXP" \
    --num_pars "$NPAR" \
    --initial_ckpt_path "$MODEL_CKPT" \
    --continual_ckpt_path "$CONTINUAL_CKPT" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --target_log_ppl "$TARGET" \
    --step_time_atol "$STEP_TIME_ATOL" \
    --ckpt_start_step "$START_STEPS" \
    --max_retries "$MAX_RETRIES" \
    $CMD_SUFFIX |& tee "$JOB_DIR/train_console.log"

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
duration=$((end - start))
duration_hms=$(printf '%02d:%02d:%02d\n' $((duration/3600)) $(( (duration%3600)/60 )) $((duration%60)))

echo "ENDING TIMING RUN AT $end_fmt"
echo "TOTAL TIMING RUN: $duration seconds (${duration_hms})"
