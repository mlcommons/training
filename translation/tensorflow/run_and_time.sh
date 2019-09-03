#/bin/bash

# This script runs preprocessing on the downloaded data
# and times (exlcusively) training to the target accuracy.


DEFAULT_SEED=1
DEFAULT_TARGET_UNCASED_BLEU_SCORE=25

set -e

usage() {
  echo \
    Usage: run_and_time.sh \
    [--random_seed RANDOM_SEED] \
    [--bleu_threshold TARGET_BLEU_SCORE] \
    [--num_gpus NUM_GPUS] \
    [--distribution_strategy DISTRIBUTION_STRATEGY] \
    [--all_reduce_alg ALL_REDUCE_ALGORITHM]
}

seed=${DEFAULT_SEED}
random_seed_arg="--random_seed=${DEFAULT_SEED}"
bleu_threshold_arg="--bleu_threshold=${DEFAULT_TARGET_UNCASED_BLEU_SCORE}"
num_gpus_arg=
distribution_strategy_arg=
all_reduce_alg_arg=

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
    --distribution_strategy )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ]; then distribution_strategy_arg="--distribution_strategy $1"; fi
      ;;
    --all_reduce_alg )
      shift
      if [[ $1 == --* ]] ; then usage; exit 1; fi
      if [ "$1" != "" ]; then all_reduce_alg_arg="--all_reduce_alg $1"; fi
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

export COMPLIANCE_FILE="/tmp/transformer_compliance_${seed}.log"
# path to the mlpef_compliance package in local directory,
# if not set then default to the package name for installing from PyPI.
export MLPERF_COMPLIANCE_PKG=${MLPERF_COMPLIANCE_PKG:-mlperf_compliance}

# Install mlperf_compliance package.
# The mlperf_compliance package is used for compliance logging.
pip3 install ${MLPERF_COMPLIANCE_PKG}

# Run preprocessing (not timed)
# TODO: Seed not currently used but will be in a future PR
. run_preprocessing.sh ${seed}

# Start timing
START=$(date +%s)
START_FMT=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${START_FMT}"

echo "Running benchmark with seed ${seed}"
. run_training.sh \
  ${random_seed_arg} \
  ${bleu_threshold_arg} \
  ${num_gpus_arg} \
  ${distribution_strategy_arg} \
  ${all_reduce_alg_arg}

RET_CODE=$?; if [[ ${RET_CODE} != 0 ]]; then exit ${RET_CODE}; fi

# End timing
END=$(date +%s)
END_FMT=$(date +%Y-%m-%d\ %r)

echo "ENDING TIMING RUN AT ${END_FMT}"

# Report result
result=$(( ${END} - ${START} ))
result_name="transformer"

echo "RESULT,${result_name},${seed},${result},${USER},${START_FMT}"
