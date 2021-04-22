#!/bin/bash

set -e

usage() {
  echo \
    Usage: run_training.sh \
    [--random_seed RANDOM_SEED] \
    [--bleu_threshold TARGET_BLEU_SCORE] \
    [--num_gpus NUM_GPUS] \
    [--distribution_strategy DISTRIBUTION_STRATEGY] \
    [--all_reduce_alg ALL_REDUCE_ALGORITHM]
}

random_seed_arg=
bleu_threshold_arg=
num_gpus_arg=
distribution_strategy_arg=
all_reduce_alg_arg=

while [ "$1" != "" ]; do
  case $1 in
    --random_seed )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ] ; then random_seed_arg="--random_seed=$1"; fi
      ;;
    --bleu_threshold )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ]; then bleu_threshold_arg="--bleu_threshold=$1"; fi
      ;;
    --num_gpus )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ]; then num_gpus_arg="--num_gpus=$1"; fi
      ;;
    --distribution_strategy )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ]; then distribution_strategy_arg="--distribution_strategy=$1"; fi
      ;;
    --all_reduce_alg )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ]; then all_reduce_alg_arg="--all_reduce_alg=$1"; fi
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

cd /research/transformer

export PYTHONPATH=/research/transformer/transformer:${PYTHONPATH}
# Add compliance to PYTHONPATH
# export PYTHONPATH=/mlperf/training/compliance:${PYTHONPATH}

python3 transformer/transformer_main.py \
  --data_dir=processed_data/ \
  --model_dir=model --params=big \
  --bleu_source=newstest2014.en \
  --bleu_ref=newstest2014.de \
  ${random_seed_arg} \
  ${bleu_threshold_arg} \
  ${num_gpus_arg} \
  ${distribution_strategy_arg} \
  ${all_reduce_alg_arg}
