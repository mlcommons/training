#/bin/bash

# This script runs preprocessing on the downloaded data
# and times (exlcusively) training to the target accuracy.

# To use the script:
# run_and_time.sh <random seed 1-5>

TARGET_UNCASED_BLEU_SCORE=25

set -e

export COMPLIANCE_FILE="/tmp/transformer_compliance_${SEED}.log"
# path to the mlpef_compliance package in local directory,
# if not set then default to the package name for installing from PyPI.
export MLPERF_COMPLIANCE_PKG=${MLPERF_COMPLIANCE_PKG:-mlperf_compliance}

# Install mlperf_compliance package.
# The mlperf_compliance package is used for compliance logging.
pip3 install ${MLPERF_COMPLIANCE_PKG}

# Run preprocessing (not timed)
# TODO: Seed not currently used but will be in a future PR
. run_preprocessing.sh ${SEED}

# Start timing
START=$(date +%s)
START_FMT=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${START_FMT}"

# Run benchmark (training)
SEED=${1:-1}

echo "Running benchmark with seed ${SEED}"
. run_training.sh ${SEED} ${TARGET_UNCASED_BLEU_SCORE}

RET_CODE=$?; if [[ ${RET_CODE} != 0 ]]; then exit ${RET_CODE}; fi

# End timing
END=$(date +%s)
END_FMT=$(date +%Y-%m-%d\ %r)

echo "ENDING TIMING RUN AT ${END_FMT}"

# Report result
result=$(( ${END} - ${START} ))
result_name="transformer"

echo "RESULT,${RESULT_NAME},${SEED},${RESULT},${USER},${START_FMT}"
