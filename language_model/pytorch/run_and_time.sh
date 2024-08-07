#!/bin/bash

SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
SLURM_JOB_ID=${SLURM_JOB_ID:-$RANDOM}
MULTI_NODE=${MULTI_NODE:-''}
echo "Run vars: id $SLURM_JOB_ID gpus $SLURM_NTASKS_PER_NODE mparams $MULTI_NODE"

# Start timing
START=$(date +%s)
START_FMT=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${START_FMT}"

BERT_CMD=${1}
SEED=${2}

# Options

set -x

eval "${BERT_CMD} --seed=${SEED}"

# End timing
END=$(date +%s)
END_FMT=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT ${END_FMT}"

# Report result
RESULT=$(( ${END} - ${START} ))
RESULT_NAME="bert"
echo "RESULT,${RESULT_NAME},${SEED},${RESULT},${USER},${START_FMT}"

set +x
