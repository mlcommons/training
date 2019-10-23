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

import logging
import os
import sys

import mllog
from mllog import constants


def dummy_example():
  """Example usage of mllog"""

  # Get the mllogger instance, this needs to be called in every module that
  # needs logging
  mllogger = mllog.get_mllogger()

  # Customize mllogger configuration
  # These configurations only need to be set Once in your entire program.
  # Try tweaking the following configurations to see the difference.
  #   logger: Customize the underlying logger to change the logging behavior.
  #   default_namespace: the default namespace to use if one isn't provided.
  #   default_stack_offset: the default depth to go into the stack to find
  #     the call site.
  #   default_clear_line: the default behavior of line clearing (i.e. print
  #     an extra new line to clear any pre-existing text in the log line).
  #   root_dir: directory prefix which will be trimmed when reporting calling
  #     file for logging.

  # Customize the underlying logger to use a file in addition to stdout.
  # You may pass a logging.Logger instance to mllog.config().
  # Notice that proper log level needs to be set for both logger and handler.
  logger = logging.getLogger("custom_logger")
  logger.setLevel(logging.DEBUG)
  # add file handler for file logging
  _file_handler = logging.FileHandler("example.log")
  _file_handler.setLevel(logging.DEBUG)
  logger.addHandler(_file_handler)
  # add stream handler for stdout logging
  _stream_handler = logging.StreamHandler(stream=sys.stdout)
  _stream_handler.setLevel(logging.INFO)
  logger.addHandler(_stream_handler)

  # Set logger configurations
  mllog.config(
      logger=logger,
      default_namespace = "worker1",
      default_stack_offset = 1,
      default_clear_line = False,
      root_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")))

  # Example log messages
  # The methods to use are "start", "end", and "event".
  # You may check out the detailed APIs in mllog.mllog.
  # Try to use the keys from mllog.constants to avoid wrong keys.
  mllogger.start(constants.RUN_START, None)
  mllogger.event(constants.GLOBAL_BATCH_SIZE, 1024)
  mllogger.event(constants.EVAL_ACCURACY, 0.99, clear_line=True)
  mllogger.end(constants.RUN_STOP, None)


if __name__ == "__main__":
  dummy_example()
