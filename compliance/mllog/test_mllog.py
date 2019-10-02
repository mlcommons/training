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

import collections
from contextlib import contextmanager
import io
import json
import os
import sys
import time
import unittest
from unittest import mock

import mllog
from mllog import constants


@contextmanager
def _captured_stdout():
  cap_out = io.StringIO()
  old_out = sys.stdout
  try:
    sys.stdout = cap_out
    yield sys.stdout
  finally:
    sys.stdout = old_out

class TestMlperfLog(unittest.TestCase):

  def setUp(self):
    self.origin_get_caller = mllog.mllog.get_caller
    mllog.mllog.get_caller = mock.MagicMock(
        return_value={"file": "mybenchmark/file.py", "lineno": 42})

    self.origin_do_log = mllog.mllog.MLLogger._do_log
    mllog.mllog.MLLogger._do_log = mock.Mock(side_effect=self._fake_do_log)

    self.origin_time = time.time
    time.time = mock.MagicMock(return_value=1234567890.123)

  def tearDown(self):
    mllog.mllog.get_caller = self.origin_get_caller
    time.time = self.origin_time

  def _fake_do_log(self, level, message, clear_line=False):
    if clear_line:
      print("\n" + message)
    else:
      print(message)

  def test_mllog_start_simple(self):
    prefix = ":::MLLOG"
    expected_log_json = json.dumps(json.loads(r'''
        {
          "namespace": "",
          "time_ms": 1234567890123,
          "event_type": "INTERVAL_START",
          "key": "run_start",
          "value": null,
          "metadata": {"file": "mybenchmark/file.py", "lineno": 42}
        }''', object_pairs_hook=collections.OrderedDict))
    expected_output = " ".join([prefix, expected_log_json])
    with _captured_stdout() as out:
      mllogger = mllog.get_mllogger()
      mllogger.start(constants.RUN_START, None)
      self.assertEqual(out.getvalue().splitlines()[0], expected_output)

  def test_mllog_end_simple(self):
    prefix = ":::MLLOG"
    expected_log_json = json.dumps(json.loads(r'''
        {
          "namespace": "",
          "time_ms": 1234567890123,
          "event_type": "INTERVAL_END",
          "key": "run_stop",
          "value": null,
          "metadata": {"file": "mybenchmark/file.py", "lineno": 42}
        }''', object_pairs_hook=collections.OrderedDict))
    expected_output = " ".join([prefix, expected_log_json])
    with _captured_stdout() as out:
      mllogger = mllog.get_mllogger()
      mllogger.end(constants.RUN_STOP, None)
      self.assertEqual(out.getvalue().splitlines()[0], expected_output)

  def test_mllog_event_simple(self):
    prefix = ":::MLLOG"
    expected_log_json = json.dumps(json.loads(r'''
        {
          "namespace": "",
          "time_ms": 1234567890123,
          "event_type": "POINT_IN_TIME",
          "key": "eval_accuracy",
          "value": 0.99,
          "metadata": {"file": "mybenchmark/file.py", "lineno": 42}
        }''', object_pairs_hook=collections.OrderedDict))
    expected_output = " ".join([prefix, expected_log_json])
    with _captured_stdout() as out:
      mllogger = mllog.get_mllogger()
      mllogger.event(constants.EVAL_ACCURACY, 0.99)
      self.assertEqual(out.getvalue().splitlines()[0], expected_output)

  def test_mllog_event_override_param(self):
    prefix = ":::MLLOG"
    expected_log_json = json.dumps(json.loads(r'''
        {
          "namespace": "worker1",
          "time_ms": 1231231230123,
          "event_type": "POINT_IN_TIME",
          "key": "eval_accuracy",
          "value": 0.99,
          "metadata": {"file": "mybenchmark/file.py", "lineno": 42}
        }''', object_pairs_hook=collections.OrderedDict))
    expected_output = "\n" + " ".join([prefix, expected_log_json]) + "\n"
    with _captured_stdout() as out:
      mllogger = mllog.get_mllogger()
      mllogger.event(constants.EVAL_ACCURACY, 0.99, namespace="worker1",
                     time_ms=1231231230123, clear_line=True)
      self.assertEqual(out.getvalue(), expected_output)


if __name__ == "__main__":
  unittest.main()
