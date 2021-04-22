#!/bin/bash

set -e

DEFAULT_SEED=1
DEFAULT_TARGET_UNCASED_BLEU_SCORE=24.00

usage() {
  echo \
    Usage: run_and_time.sh \
    [--random_seed RANDOM_SEED] \
    [--bleu_threshold TARGET_BLEU_SCORE] \
    [--num_gpus NUM_GPUS]
}

seed=${DEFAULT_SEED}
random_seed_arg="--random_seed ${DEFAULT_SEED}"
bleu_threshold_arg="--bleu_threshold ${DEFAULT_TARGET_UNCASED_BLEU_SCORE}"
num_gpus_arg=

while [ "$1" != "" ]; do
  case $1 in
    --random_seed )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ] ; then random_seed_arg="--random_seed $1"; seed=$1; fi
      ;;
    --bleu_threshold )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ]; then bleu_threshold_arg="--bleu_threshold $1"; fi
      ;;
    --num_gpus )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ]; then num_gpus_arg="--num_gpus $1"; fi
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

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

echo "running benchmark"
./run.sh \
  ${random_seed_arg} \
  ${bleu_threshold_arg} \
  ${num_gpus_arg}

sleep 3
ret_code=$?; if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="RNN_TRANSLATOR"

echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
