#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

set -euxo pipefail

# Change directory to the model directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR/..

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
: "${DATADIR:?DATADIR not set}"
: "${MODEL:?MODEL not set}"
: "${TOKENIZER:?TOKENIZER not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=5.0.0}"
: "${LOGDIR:=./results}"
: "${DEPENDENCIES:=./dependencies}"
: "${CONT_NAME:=dev-${CUSTOM_TAG}}"
: "${LOG_FREQ:=0}"

# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name="${CONT_NAME}"
_cont_mounts=("--volume=${DATADIR}:/data/data" "--volume=${MODEL}:/data/model/" "--volume=${TOKENIZER}:/data/tokenizer/" "--volume=$(pwd):/workspace/code"  "--volume=$(pwd)/../../AMD:/workspace/AMD" "--volume=$(pwd)/../../utilities:/workspace/utilities" "--volume=${LOGDIR}:/results")


# Setup directories
mkdir -p "${LOGDIR}"
mkdir -p "${LOGDIR}/artifacts/"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(MODEL)
_config_env+=(DGXSYSTEM)
_config_env+=(LOGDIR)

echo "TEST"
echo ${_config_env[@]}
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${_cont_name}$"; then
        docker container rm -f "${_cont_name}" || true
    else
        echo "Container ${_cont_name} does not exist. Skipping removal."
    fi
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

docker run --rm --init --detach \
      --net=host --uts=host \
      --ipc=host --device /dev/dri --device /dev/kfd \
      --security-opt=seccomp=unconfined \
      --name="${_cont_name}" "${_cont_mounts[@]}" \
      -e IMAGE_NAME="${CONT}" \
      "${CONT}" sleep infinity

# Make sure container has time to finish initialization
sleep 5
# bash runtime_tunables.sh
docker exec "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
  (
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    if [[ $CLEAR_CACHES == 1 ]]; then
      bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
    fi
    _config_env+=(--env=SEED=$RANDOM) # Reset random seed
    echo 'launching experiment using:'  ${_config_env[@]} ${_cont_name} ./dev/run_llama31.sh
    docker exec  ${_config_env[@]} --env=HYDRA_FULL_ERROR ${_cont_name} ./dev/run_llama31.sh
  ) | tee "${_logfile_base}_${_experiment_index}.log"

    if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec "${_config_env[@]}" "${_cont_name}"  \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log" \
    || true
    fi

done

echo "Number of experiments $NEXP"
