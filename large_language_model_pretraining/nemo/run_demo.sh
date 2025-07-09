#!/bin/bash

set +x
set -e

source demo_config.sh

git init
git add .
git config --global user.name "test"
git config --global user.email "test@test.com"
git commit -m "start"

# Vars without defaults
#     Slurm settings
: "${USER:?USER not set}"
: "${HOST:?HOST not set}"
: "${ACCOUNT:?ACCOUNT not set}"
: "${PARTITION:?PARTITION not set}"

#     Job settings
: "${IMAGE:?IMAGE not set}"

#     Dataset settings
#: "${PREPROCESSED_PATH:?PREPROCESSED_PATH not set}"
#: "${TOKENIZER_PATH:?TOKENIZER_PATH not set}"

#     Model settings
#: "${MODEL_CKPT:?MODEL_CKPT not set}"
: "${USE_CKPT:?USE_CKPT not set}"
: "${FROM_HF:?FROM_HF not set}"
: "${CONTINUAL_CKPT:?CONTINUAL_CKPT not set}"

# Vars with defaults
#     Slurm settings
: "${TIME:="04:00:00"}"
: "${NNODES:=288}"
: "${GPUS_PER_NODE:=8}"
: "${DEPENDENCIES:=""}"

#     Job settings
: "${NEMO_DIR:=""}" # Provide customized NeMo path here
: "${NEMO_RUN_DIR:=""}" # Provide customized NeMo-Run path here
: "${TMP_NPY_INDEX:=""}" # Provide temporary NNumpy Index saving directory
: "${MAX_RETRIES:=0}"

#     Model settings
: "${SIZE:="405b"}"
: "${GBS:=1152}"
: "${MBS:=1}"
: "${START_STEPS:=0}"

#     Dataloader settings
: "${MAX_STEPS:=""}"

#     Experiment settings
: "${SEEDS:=""}"
IFS=" " read -ra seeds <<< $SEEDS
: "${NEXP:=1}"
: "${NPAR:=1}"
: "${SAVE_CKPT:=0}"
: "${TAG:=""}"
: "${TARGET:="5.6"}"
: "${STEP_TIME_ATOL:="7200"}" # maximum tolerable step time, setting to 2hr by default

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


# Run

MOUNTS="${JOB_DIR}:/output,${JOB_DIR}:/mlperf-outputs,${PREPROCESSED_PATH}:/preproc_data,${MODEL_CKPT}:/checkpoint,${TOKENIZER_PATH}:/tokenizer,${CONTINUAL_CKPT}:/continual"

CKPT_OPTION=""

CMD_SUFFIX=""

if [ $USE_CKPT -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --use_ckpt"
    if [ $FROM_HF -gt 0 ]; then
        CMD_SUFFIX="${CMD_SUFFIX} --resume_from_hf"
    fi
fi

if [ $SAVE_CKPT -gt 0 ]; then 
    CMD_SUFFIX="${CMD_SUFFIX} --save_ckpt"
fi

if [ ! $NEMO_DIR = "" ]; then
    MOUNTS="${MOUNTS},${NEMO_DIR}:/opt/NeMo"
fi

if [ ! $NEMO_RUN_DIR = "" ]; then
    MOUNTS="${MOUNTS},${NEMO_RUN_DIR}:/opt/NeMo-Run"
fi

if [ ! $TMP_NPY_INDEX = "" ]; then
    MOUNTS="${MOUNTS},${TMP_NPY_INDEX}:/npy_index"
fi

if [ ! $DEPENDENCIES = "" ]; then 
    CMD_SUFFIX="${CMD_SUFFIX} --dependencies ${DEPENDENCIES}"
fi

if [ ! $MAX_STEPS = "" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --max_steps ${MAX_STEPS}"
fi

if [ ! $TAG = "" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --tag ${TAG}"
fi

# Allows MLLogger objects to be constructed locally
if [ ! -d /mlperf-outputs ]; then mkdir /mlperf-outputs; fi

set -x

python3 pretrain_llama31.py \
--user $USER --host $HOST \
--job_dir $JOB_DIR \
--account $ACCOUNT --partition $PARTITION \
--nodes $NNODES --gpus_per_node $GPUS_PER_NODE \
--time $TIME \
--mounts $MOUNTS \
--image $IMAGE \
--size $SIZE \
--gbs $GBS --mbs $MBS \
--seeds ${seeds[@]} \
--num_exps $NEXP \
--num_pars $NPAR \
--initial_ckpt_path /checkpoint \
--continual_ckpt_path /continual \
--tokenizer_path /tokenizer \
--target_log_ppl $TARGET \
--step_time_atol $STEP_TIME_ATOL \
--ckpt_start_step $START_STEPS \
--max_retries $MAX_RETRIES \
$CMD_SUFFIX |& tee "$JOB_DIR/train_console.log"

# End timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
duration=$((end - start))
duration_hms=$(printf '%02d:%02d:%02d\n' $((duration/3600)) $(( (duration%3600)/60 )) $((duration%60)))

echo "ENDING TIMING RUN AT $end_fmt"
echo "TOTAL TIMING RUN: $duration seconds (${duration_hms})"
