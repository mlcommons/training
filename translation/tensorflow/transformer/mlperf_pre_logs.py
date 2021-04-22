# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
# ==============================================================================

"""Log MLPerf required information outside benchmark run."""
# NOTE: This script logs a list of "fake" information so that the logs generated
# by the reference codes can pass the MLPerf compliance checking. In real MLPerf
# submissions, the corresponding information should be logged with "real" values
# and at proper locations that reflect the reality.
# Run this script before running the reference training codes.

import os

from mlperf_logging import mllog
from mlperf_logging.mllog import constants as mllog_const


mllogger = mllog.get_mllogger()

def main():
  mllog.config(
    filename=(os.getenv("COMPLIANCE_FILE") or "mlperf_compliance.log"),
    root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__)))
  )

  mllogger.event(key=mllog_const.SUBMISSION_BENCHMARK, value="transformer")
  mllogger.event(key=mllog_const.SUBMISSION_DIVISION, value="closed")
  mllogger.event(key=mllog_const.SUBMISSION_ORG, value="ref")
  mllogger.event(key=mllog_const.SUBMISSION_PLATFORM, value="ref")
  mllogger.event(key=mllog_const.SUBMISSION_STATUS, value="research")

  mllogger.event(key=mllog_const.CACHE_CLEAR, value=True)


if __name__ == "__main__":
  main()
