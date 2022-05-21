#!/bin/bash

# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set +x
set -e

# Only rank print
[ "${SLURM_LOCALID-}" -ne 0 ] && set +x


# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Set variables
[ "${DEBUG}" = "1" ] && set -x
BATCHSIZE=${BATCHSIZE:-2}
EVALBATCHSIZE=${EVALBATCHSIZE:-${BATCHSIZE}}
NUMEPOCHS=${NUMEPOCHS:-30}
LOG_INTERVAL=${LOG_INTERVAL:-20}
DATASET_DIR=${DATASET_DIR:-"/datasets/open-images-v6-mlperf"}
TORCH_HOME=${TORCH_HOME:-"$(pwd)/torch-model-cache"}

# run benchmark
echo "running benchmark"



declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
    cluster=''
    if [[ "${DGXSYSTEM}" == DGX2* ]]; then
        cluster='circe'
    fi
    if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
        cluster='selene'
    fi
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
    CMD=( './bind.sh' "--cluster=${cluster}" '--ib=single' '--' ${NSYSCMD} 'python' '-u' )
  else
    CMD=( 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker; need to launch tasks with torchrun
  CMD=( "torchrun" "--standalone" "--nnodes=1" "--nproc_per_node=${DGXNGPU}" )
  [ "$MEMBIND" = false ] &&  CMD+=( "--no_membind" )
fi

PARAMS=(
      --batch-size              "${BATCHSIZE}"
      --eval-batch-size         "${EVALBATCHSIZE}"
      --epochs                  "${NUMEPOCHS}"
      --print-freq              "${LOG_INTERVAL}"
      --data-path               "${DATASET_DIR}"
)

# run training
"${CMD[@]}" train.py "${PARAMS[@]}" ${EXTRA_PARAMS} ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="SINGLE_STAGE_DETECTOR"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
