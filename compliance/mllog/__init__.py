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
import sys
import threading

from mllog import mllog


# _lock is for serializing access to shared data structures in this module.
_lock = threading.RLock()

# mllogger is a logger shareable across modules.
mllogger = mllog.MLLogger()


def get_mllogger():
  """Get the shared logger."""
  return mllogger

def config(**kwargs):
  """Configure the shared logger.
  Optional keyword arguments:
    logger: a logging.Logger instance. Customize the logger to change
      the logging behavior (e.g. logging to a file, etc.)
    filename: a log file to use. If set, a default file handler will be added
      to the logger so it can log to the specified file. For more advanced
      customizations, please set the 'logger' parameter instead.
    default_namespace: the default namespace to use if one isn't provided.
    default_stack_offset: the default depth to go into the stack to find the
      call site.
    default_clear_line: the default behavior of line clearing (i.e. print
      an extra new line to clear any pre-existing text in the log line).
    root_dir: directory prefix which will be trimmed when reporting calling
      file for logging.
  """
  if _lock:
    _lock.acquire()
  try:
    logger = kwargs.pop("logger", None)
    if logger is not None:
      if not isinstance(logger, logging.Logger):
        raise ValueError("'logger' must be an instance of 'logging.Logger'.")
      if logger.name == mllogger.logger.name:
        raise ValueError("'logger' should not be the same as the default " +
                         "logger to avoid unexpected behavior. Consider " +
                         "using a different name for the logger.")
      mllogger.logger = logger

    log_file = kwargs.pop("filename", None)
    if log_file is not None:
      if not isinstance(log_file, str):
        raise ValueError("'filename' must be a string.")
      _file_handler = logging.FileHandler(log_file)
      _file_handler.setLevel(logging.INFO)
      mllogger.logger.addHandler(_file_handler)

    default_namespace = kwargs.pop("default_namespace", None)
    if default_namespace is not None:
      if not isinstance(default_namespace, str):
        raise ValueError("'default_namespace' must be a string.")
      mllogger.default_namespace = default_namespace

    default_stack_offset = kwargs.pop("default_stack_offset", None)
    if default_stack_offset is not None:
      if not isinstance(default_stack_offset, int):
        raise ValueError("'default_stack_offset' must be an integer.")
      mllogger.default_stack_offset = default_stack_offset

    default_clear_line = kwargs.pop("default_clear_line", None)
    if default_clear_line is not None:
      if not isinstance(default_clear_line, bool):
        raise ValueError("'default_clear_line' must be a boolean value.")
      mllogger.default_clear_line = default_clear_line

    root_dir = kwargs.pop("root_dir", None)
    if root_dir is not None:
      if not isinstance(root_dir, str):
        raise ValueError("'root_dir' must be a string.")
      mllogger.root_dir = root_dir

  finally:
    if _lock:
      _lock.release()
