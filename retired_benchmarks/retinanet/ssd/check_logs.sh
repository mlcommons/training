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

LOG_DIR=${LOG_DIR:-"./logs/"}

# Handle MLCube parameters
while [ $# -gt 0 ]; do
  case "$1" in
    --log_dir=*)
      LOG_DIR="${1#*=}"
      ;;
    --checker_logs_dir=*)
      CHECKER_LOG_DIR="${1#*=}"
      ;;
    *)
  esac
  shift
done

for filename in $LOG_DIR/*.log; do
    log_file=${filename##*/}
    python -m mlperf_logging.compliance_checker $filename --log_output $CHECKER_LOG_DIR/$log_file || true
done
