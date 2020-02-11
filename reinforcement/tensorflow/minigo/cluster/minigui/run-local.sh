#!/bin/bash
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run in a sub-shell so we don't unexpectedly set new variables.

# The model directory stuff is confusing. Try to help the user. Here they've
# set the board size but not the GCS DIR. So do some guessing before we set
# defaults
if [[ ! -z "${MINIGUI_BOARD_SIZE}" && -z "${MINIGUI_GCS_DIR}" ]]; then
  if [[ "${MINIGUI_BOARD_SIZE}" = "19" ]];  then
    export MINIGUI_GCS_DIR="v5-19x19/models"
  fi
  if [[ "${MINIGUI_BOARD_SIZE}" = "9" ]];  then
    export MINIGUI_GCS_DIR="v3-9x9/models"
  fi
fi


source ${SCRIPT_DIR}/../../minigui/minigui-common.sh
source ${SCRIPT_DIR}/../common.sh

echo "Using: the following defaults for run-local:"
echo "--------------------------------------------------"
echo "MINIGUI_MODEL:        ${MINIGUI_MODEL}"
echo "MINIGUI_MODEL_TMPDIR: ${MINIGUI_MODEL_TMPDIR}"
echo "MINIGUI_PORT:         ${MINIGUI_PORT}"
echo "MINIGUI_GCS_DIR:      ${MINIGUI_GCS_DIR}"
echo "MINIGUI_BOARD_SIZE:   ${MINIGUI_BOARD_SIZE}"

echo "PROJECT:              ${PROJECT}"
echo "VERSION_TAG:          ${VERSION_TAG}"
echo "MINIGUI CONTAINER:    ${MINIGUI_PY_CPU_CONTAINER}"
echo "IMAGE                 gcr.io/${PROJECT}/${MINIGUI_PY_CPU_CONTAINER}:${VERSION_TAG}"
echo

if [[ -d "${MINIGUI_MODEL_TMPDIR}" ]]; then
  docker run \
  -p 127.0.0.1:$MINIGUI_PORT:$MINIGUI_PORT \
  -e MINIGUI_MODEL="${MINIGUI_MODEL}" \
  -e MINIGUI_BOARD_SIZE="${MINIGUI_BOARD_SIZE}" \
  -e MINIGUI_GCS_DIR="${MINIGUI_GCS_DIR}" \
  -ti \
  --mount type=bind,source="${MINIGUI_MODEL_TMPDIR}",target="${MINIGUI_MODEL_TMPDIR}" \
  --rm gcr.io/${PROJECT}/${MINIGUI_PY_CPU_CONTAINER}:${VERSION_TAG}
else
  docker run \
  -p 127.0.0.1:$MINIGUI_PORT:$MINIGUI_PORT \
  -e MINIGUI_MODEL="${MINIGUI_MODEL}" \
  -e MINIGUI_BOARD_SIZE="${MINIGUI_BOARD_SIZE}" \
  -e MINIGUI_GCS_DIR="${MINIGUI_GCS_DIR}" \
  -ti \
  --rm gcr.io/${PROJECT}/${MINIGUI_PY_CPU_CONTAINER}:${VERSION_TAG}
fi
