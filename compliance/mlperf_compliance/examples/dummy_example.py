# Copyright 2019 MLBenchmark Group. All Rights Reserved.
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

import os

from mlperf_compliance import constants
from mlperf_compliance import mlperf_log

def run_example_logs():
  """Example usage of mlperf_compliance."""

  # Find root dir of the benchmark, this is for this example only,
  # you may apply a different approach to find root dir.
  root_dir = os.path.normpath(
      os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

  # Set default variables, the default variables will be applied to all logs.
  # You can try changing some of the default values and see their effects.
  # The following information can be set:
  #   benchmark: benchmark name, this is mainly used for model-specific
  #       validations, not setting benchmark name will not cause errors.
  #   root_dir: the root directory of benchmark, used for printing file paths.
  #   stack_offset: (default=1) increase the value to go deeper into the stack
  #       to find the callsite. For example, if this is being called by a
  #       wraper/helper you may want to set stack_offset=1 to use the callsite
  #       of the wraper/helper itself.
  #   extra_print: (default=False) print a blank line before logging to clear
  #       any text in the line.
  #   prefix: (default="") string with which to prefix the log message. Useful
  #       for differentiating raw lines if stitching will be required.
  mlperf_log.setdefault(
      benchmark="resnet",
      root_dir=root_dir,
      stack_offset=1,
      extra_print=False,
      prefix="")

  # These logs will print as expected
  mlperf_log.mlperf_print(constants.RUN_START)
  mlperf_log.mlperf_print(constants.EVAL_ACCURACY, value=0.99)
  mlperf_log.mlperf_print(
          constants.EPOCH_STOP, metadata={"first_epoch_num": 1})
  
  # These logs may cause a warning message.
  # The warnings are for reference only, they might not be up-to-date with
  # the latest logging policy, and might not cover all requirements.
  # Currently, only keys are checked against known key list if benchmark
  # name is set. If you intend to log a new key that is not known in the
  # policy doc, please ignore the warning.
  mlperf_log.mlperf_print("unknown_key")


if __name__ == "__main__":
  run_example_logs()
