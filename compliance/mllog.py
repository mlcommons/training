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

_PREFIX = ':::MLLOG'
PATTERN = re.compile('[a-zA-Z0-9]+')

LOG_FILE = os.getenv('COMPLIANCE_FILE')
LOGGER = logging.getLogger('mlperf_compliance')
LOGGER.setLevel(logging.DEBUG)

_STREAM_HANDLER = logging.StreamHandler(stream=sys.stdout)
_STREAM_HANDLER.setLevel(logging.INFO)
LOGGER.addHandler(_STREAM_HANDLER)

if LOG_FILE:
  _FILE_HANDLER = logging.FileHandler(LOG_FILE)
  _FILE_HANDLER.setLevel(logging.DEBUG)
  LOGGER.addHandler(_FILE_HANDLER)
else:
  _STREAM_HANDLER.setLevel(logging.DEBUG)


def get_caller(stack_index=2, root_dir=None):
  caller = inspect.getframeinfo(inspect.stack()[stack_index][0])

  # Trim the filenames for readability.
  filename = caller.filename
  if root_dir is not None:
    filename = re.sub('^' + root_dir + '/', '', filename)
  return (filename, caller.lineno)


# :::MLL 1556733699.71 run_start: {"value": null,
# "metadata": {"lineno": 77, "file": main.py}}
LOG_TEMPLATE = ':::MLL {:.3f} {}: {{"value": {}, "metadata": {}}}'


def _now_as_str():
  """Returns the current time as a human readable string."""
  return datetime.datetime.now().strftime('%H:%M:%S.%f')


def _current_time_ns():
  """Returns current nanoseconds since epoch."""
  return time.time() * 1000 * 1000 * 1000


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


def _encode_log(namespace, time_ns, event_type, log_key, log_value):
  """Encodes an MLEvent as a string log line.
  Args:
    namespace: provides structure, e.g. "GPU0".
    time_ns: nano seconds since unix epoch.
    event_type: one of: 'INTERVAL_START', 'INTERVAL_END', 'POINT_IN_TIME'
    log_key: the name of the thing being logged.
    log_value: a json value.
  Returns:
    A string log like, i.e. ":::MLLog { ..."
  """
  json_val = {
      'namespace': str(namespace),
      'time_ns': int(time_ns),
      'event_type': str(event_type),
      'key': str(log_key),
      'value': str(log_value)
  }
  encoded = _to_ordered_json(json_val.items())
  return '{} {}'.format(_PREFIX, encoded)


class MLLogger(object):
  """Creates a new logger.
  This allows logging of MLEvents through stdout.
  """

  def __init__(self, default_namespace='', log_info=True, log_file=None):
    """Create a new logger.
    Args:
      default_namespace: the default namespace to use if one isn't provided.
      log_info: True if log lines should be printed to LOGGER.info()
      log_file: an optional file object which to also write loglines to.
    """
    self.default_namespace = default_namespace
    self.log_info = log_info
    self.log_file = log_file

  def _log_helper(self, log_key, log_value, namespace, event_type,
                  time_ns=None):
    """Log an event."""
    if namespace is None:
      namespace = self.default_namespace
    if time_ns is None:
      time_ns = _current_time_ns()

    log_line = _encode_log(
        namespace,
        time_ns,
        event_type,
        log_key,
        log_value)

    if self.log_info:
      LOGGER.info(log_line)
    if self.log_file:
      self.log_file.write(log_line + '\n')

  def start(self, log_key, log_value, namespace=None, time_ns=None):
    """Start an time interval in the log.
    All intervals which are started must be ended. This interval must be
    ended before a new interval with the same key and namespace can be started.
    Args:
      log_key: the key for the event, e.g. "mlperf.training"
      log_value: the json value to log.
      namespace: override the default namespace.
      time_ns: the time in nanoseconds, or None for current time.
    """
    self._log_helper(log_key, log_value, namespace, 'INTERVAL_START',
                     time_ns=time_ns)

  def end(self, log_key, log_value, namespace=None, time_ns=None):
    """End a time interval in the log.
    Ends an interval which was already started with the same key and in the
    same namespace.
    Args:
      log_key: the same log key which was passed to start().
      log_value: the value to log at the end of the interval.
      namespace: optional override of the default namespace.
      time_ns: the time in nanoseconds, or None for current time.
    """
    self._log_helper(log_key, log_value, namespace, 'INTERVAL_END',
                     time_ns=time_ns)

  def event(self, log_key, log_value, namespace=None, time_ns=None):
    """Log a point in time event.
    The event does not have an associated duration like an interval has.
    Args:
      log_key: the "name" of the event.
      log_value: the event data itself.
      namespace: optional override of the default namespace.
      time_ns: the time in nanoseconds, or None for current time.
    """
    self._log_helper(log_key, log_value, namespace, 'POINT_IN_TIME',
                     time_ns=time_ns)
