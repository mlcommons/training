#!/bin/bash

set -e

DATASET_DIR='/data'

usage() {
  echo \
    Usage: run.sh \
    [--random_seed RANDOM_SEED] \
    [--bleu_threshold TARGET_BLEU_SCORE] \
    [--num_gpus NUM_GPUS]
}

SEED=1
TARGET=24.00
MULTI_GPU_WRAPPER=

MULTI_GPU_COMMAND='-m torch.distributed.launch --nproc_per_node'

while [ "$1" != "" ]; do
  case $1 in
    --random_seed )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ] ; then SEED=$1; fi
      ;;
    --bleu_threshold )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ]; then TARGET=$1; fi
      ;;
    --num_gpus )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ] && [ "$1" != "1" ] ; then MULTI_GPU_WRAPPER="$MULTI_GPU_COMMAND $1"; fi
      ;;
    -h | --help )
      usage
      exit
      ;;
    * )
      usage
      exit 1
  esac
  shift
done

# clear your caches here
python -c \
'from seq2seq.utils import gnmt_event;'\
'from mlperf_logging.mllog import constants;'\
'gnmt_event(constants.CACHE_CLEAR, value=True)'

# run training
python3 $MULTI_GPU_WRAPPER train.py \
  --dataset-dir ${DATASET_DIR} \
  --seed $SEED \
  --train-global-batch-size 1024 \
  --target-bleu $TARGET
