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
"""Convenience function for logging compliance tags to stdout."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import inspect
import json
import logging
import os
import re
import sys
import time
import timeit

from mllog import constants

LOG_TEMPLATE = ':::MLLOG {log_json}'


def get_caller(stack_index=2, root_dir=None):
  """Get caller's file and line number information.
  Args:
    stack_index: a stack_index of 2 will provide the caller of the
      function calling this function. Notice that stack_index of 2
      or more will fail if called from global scope.
    root_dir: the root dir prefixed to the file name. The root_dir
      will be trimmed from the file name in the output.
  Returns:
    Call site info in a dictionary with these fields:
      "file": (string) file path
      "lineno": (int) line number
  """
  caller = inspect.getframeinfo(inspect.stack()[stack_index][0])

  # Trim the filenames for readability.
  filename = caller.filename
  if root_dir is not None:
    filename = re.sub('^' + root_dir + '/', '', filename)
  return {"file": filename, "lineno": caller.lineno}


def _now_as_str():
  """Returns the current time as a human readable string."""
  return datetime.datetime.now().strftime('%H:%M:%S.%f')


def _current_time_ms():
  """Returns current milliseconds since epoch."""
  return int(time.time() * 1e3)


def _try_float(value):
  """Tries to convert a value to a float (otherwise returns the input)."""
  try:
    return float(value)
  except Exception:  # pylint:disable=broad-except
    return value


def _to_ordered_json(kv_pairs):
  """Convert a list of (key, value) pairs to a serialized json message.
  Args:
    kv_pairs: List of (key, value) tuples, ordered by how they should be
      presented.
  Returns:
    string - The serialized json string with fields in the same order as
      kv_pairs. If there were any key/value pairs that couldn't be properly
      converted to json, returns an error string.
  """
  d = collections.OrderedDict()
  # json.dumps() works differently in python 2 and 3 for boolean keys.
  # py2: json.dumps({True: 'a', False: 'b'}) -> '{"True": "a", "False": "b"}'
  # py3: json.dumps({True: 'a', False: 'b'}) -> '{"true": "a", "false": "b"}'
  # We standardize to match python 3 output.
  # Note that we don't need to account for this in boolean values.
  # py2 and 3: json.dumps({'a': True, 'b': False}) -> '{"a": true, "b": false}'
  for key, value in kv_pairs:
    if isinstance(key, bool):
      if key:
        d['true'] = value
      else:
        d['false'] = value
    elif key == 'value':
      # See if we can convert the 'value' field to a float where possible.
      # This is to mostly handle np.float values which cannot be json encoded.
      d[key] = _try_float(value)
    else:
      d[key] = value
  try:
    return json.dumps(d)
  except Exception:  # pylint:disable=broad-except
    return '[convert-error: {kv_str}]'.format(kv_str=str(kv_pairs))


def _encode_log(namespace, time_ms, event_type, key, value, call_site):
  """Encodes an MLEvent as a string log line.
  Args:
    namespace: provides structure, e.g. "GPU0".
    time_ms: milliseconds since unix epoch.
    event_type: one of: 'INTERVAL_START', 'INTERVAL_END', 'POINT_IN_TIME'
    key: the name of the thing being logged.
    value: a json value.
    call_site: a json pointing to the source code logging the event;
        (e.g {"file": "train.py", "lineno": 42})
  Returns:
    A string log like, i.e. ":::MLLog { ..."
  """
  # preserve the order of key-values
  ordered_key_val_pairs = [
    ('namespace', namespace),
    ('time_ms', time_ms),
    ('event_type', event_type),
    ('key', key),
    ('value', value),
    ('call_site', call_site)
  ]
  encoded = _to_ordered_json(ordered_key_val_pairs)
  return LOG_TEMPLATE.format(log_json=encoded)


class MLLogger(object):
  """MLPerf logging helper."""

  def __init__(self,
               logger=None,
               default_namespace=constants.DEFAULT_NAMESPACE,
               default_stack_offset=1,
               default_clear_line=False,
               root_dir=None):
    """Create a new MLLogger.
    Args:
      logger: a logging.Logger instance. If not specified, a default logger
        will be used which prints to stdout. Customize the logger to change
        the logging behavior (e.g. logging to a file, etc.)
      default_namespace: the default namespace to use if one isn't provided.
      default_stack_offset: the default depth to go into the stack to find the
        call site. Default value is 1.
      default_clear_line: the default behavior of line clearing (i.e. print
        an extra new line to clear any pre-existing text in the log line).
      root_dir: directory prefix which will be trimmed when reporting calling
        file for logging.
    """
    if logger is None:
      self.logger = self._get_default_logger()
    elif not isinstance(logger, logging.Logger):
      raise ValueError("logger must be a `logging.Logger` instance.")
    else:
      self.logger = logger

    self.default_namespace = default_namespace
    self.default_stack_offset = default_stack_offset
    self.default_clear_line = default_clear_line
    self.root_dir = root_dir
  
  def _get_default_logger(self):
    """Create a default logger.
    The default logger prints INFO level messages to stdout.
    """
    logger = logging.getLogger(constants.DEFAULT_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    _stream_handler = logging.StreamHandler(stream=sys.stdout)
    _stream_handler.setLevel(logging.INFO)
    logger.addHandler(_stream_handler)
    return logger

  def _do_log(self, message, clear_line=False):
    if clear_line:
      message = '\n' + message
    self.logger.info(message)

  def _log_helper(self, event_type, key, value, namespace=None, time_ms=None,
                  stack_offset=None, clear_line=None):
    """Log an event."""
    if namespace is None:
      namespace = self.default_namespace
    if time_ms is None:
      time_ms = _current_time_ms()
    if stack_offset is None:
      stack_offset = self.default_stack_offset
    if clear_line is None:
      clear_line = self.default_clear_line

    call_site = get_caller(2 + stack_offset, root_dir=self.root_dir)

    log_line = _encode_log(
        namespace,
        time_ms,
        event_type,
        key,
        value,
        call_site)
    
    self._do_log(log_line, clear_line)

  def start(self, key, value, namespace=None, time_ms=None,
            stack_offset=None, clear_line=None):
    """Start an time interval in the log.
    All intervals which are started must be ended. This interval must be
    ended before a new interval with the same key and namespace can be started.
    Args:
      key: the key for the event, e.g. "mlperf.training"
      value: the json value to log.
      namespace: override the default namespace.
      time_ms: the time in milliseconds, or None for current time.
      stack_offset: override the default stack offset, i.e. the depth to go
        into the stack to find the call site.
      clear_line: override the default line clearing behavior, i.e. whether to
        print an extra new line to clear pre-existing text in the log line.
    """
    self._log_helper(constants.INTERVAL_START, key, value,
                     namespace=namespace, time_ms=time_ms,
                     stack_offset=stack_offset, clear_line=clear_line)

  def end(self, key, value, namespace=None, time_ms=None,
          stack_offset=None, clear_line=None):
    """End a time interval in the log.
    Ends an interval which was already started with the same key and in the
    same namespace.
    Args:
      key: the same log key which was passed to start().
      value: the value to log at the end of the interval.
      namespace: optional override of the default namespace.
      time_ms: the time in milliseconds, or None for current time.
      stack_offset: override the default stack offset, i.e. the depth to go
        into the stack to find the call site.
      clear_line: override the default line clearing behavior, i.e. whether to
        print an extra new line to clear pre-existing text in the log line.
    """
    self._log_helper(constants.INTERVAL_END, key, value,
                     namespace=namespace, time_ms=time_ms,
                     stack_offset=stack_offset, clear_line=clear_line)

  def event(self, key, value, namespace=None, time_ms=None,
            stack_offset=None, clear_line=None):
    """Log a point in time event.
    The event does not have an associated duration like an interval has.
    Args:
      key: the "name" of the event.
      value: the event data itself.
      namespace: optional override of the default namespace.
      time_ms: the time in milliseconds, or None for current time.
      stack_offset: override the default stack offset, i.e. the depth to go
        into the stack to find the call site.
      clear_line: override the default line clearing behavior, i.e. whether to
        print an extra new line to clear pre-existing text in the log line.
    """
    self._log_helper(constants.POINT_IN_TIME, key, value,
                     namespace=namespace, time_ms=time_ms,
                     stack_offset=stack_offset, clear_line=clear_line)
