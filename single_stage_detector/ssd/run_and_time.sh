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

# run benchmark
echo "running benchmark"

export TORCH_HOME="$(pwd)/torch-model-cache"

declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
    CMD=( './bind.sh' '--' 'python' '-u' )
  else
    CMD=( 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker; need to launch tasks with Pytorch's distributed launch
  # TODO: use bind.sh instead of bind_launch.py
  #       torch.distributed.launch only accepts Python programs (not bash scripts) to exec
  CMD=( "python" "-u" "-m" "bind_launch" "--nsockets_per_node=${DGXNSOCKET}" \
        "--ncores_per_socket=${DGXSOCKETCORES}" "--nproc_per_node=${DGXNGPU}" )
  [ "$MEMBIND" = false ] &&  CMD+=( "--no_membind" )
fi



"${CMD[@]}" $TRAIN_CMD

ret_code=$?


sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="RetinaNet_Object_Detection"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
