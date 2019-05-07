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

from contextlib import contextmanager
import io
import json
import os
import sys
import time
import unittest
from unittest import mock

from mlperf_compliance import constants
from mlperf_compliance import mlperf_log
from mlperf_compliance import validations

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
    self.origin_validate = validations.validate
    validations.validate = mock.MagicMock(return_value=True)

    self.origin_get_caller = mlperf_log.get_caller
    mlperf_log.get_caller = mock.MagicMock(
        return_value={"file": "mybenchmark/file.py", "lineno": 42})

    self.origin_do_print = mlperf_log._do_print
    mlperf_log._do_print = mock.Mock(side_effect=self._fake_do_print)

    self.origin_time = time.time
    time.time = mock.MagicMock(return_value=1558767599.999)

  def tearDown(self):
    validations.validate = self.origin_validate
    mlperf_log.get_caller = self.origin_get_caller
    mlperf_log._do_print = self.origin_do_print
    time.time = self.origin_time

  def _fake_do_print(self, key, message, extra_print=False):
    print(message)

  def test_mlperf_print_simple(self):
    expected_output_l = ":::MLL 1558767599.999 run_start:"
    expected_output_r = '{"value": null, ' + \
        '"metadata": {"file": "mybenchmark/file.py", "lineno": 42}}'
    with _captured_stdout() as out:
      mlperf_log.mlperf_print(constants.RUN_START)
      lines = out.getvalue().splitlines()
      output_l = " ".join(lines[0].split(" ", 3)[0:3])
      output_r = lines[0].split(" ", 3)[3]
      self.assertEqual(output_l, expected_output_l)
      self.assertDictEqual(json.loads(output_r), json.loads(expected_output_r))

  def test_mlperf_print_with_value(self):
    expected_output_l = ":::MLL 1558767599.999 eval_accuracy:"
    expected_output_r = '{"value": 0.99, ' + \
        '"metadata": {"file": "mybenchmark/file.py", "lineno": 42}}'
    with _captured_stdout() as out:
      mlperf_log.mlperf_print(constants.EVAL_ACCURACY, value=0.99)
      lines = out.getvalue().splitlines()
      output_l = " ".join(lines[0].split(" ", 3)[0:3])
      output_r = lines[0].split(" ", 3)[3]
      self.assertEqual(output_l, expected_output_l)
      self.assertDictEqual(json.loads(output_r), json.loads(expected_output_r))

  def test_mlperf_print_with_metadata(self):
    expected_output_l = ":::MLL 1558767599.999 epoch_stop:"
    expected_output_r = '{"value": null, "metadata": ' + \
        '{"file": "mybenchmark/file.py", "lineno": 42, "first_epoch_num": 1}}'
    with _captured_stdout() as out:
      mlperf_log.mlperf_print(
          constants.EPOCH_STOP, metadata={"first_epoch_num": 1})
      lines = out.getvalue().splitlines()
      output_l = " ".join(lines[0].split(" ", 3)[0:3])
      output_r = lines[0].split(" ", 3)[3]
      self.assertEqual(output_l, expected_output_l)
      self.assertDictEqual(json.loads(output_r), json.loads(expected_output_r))

def manual_test():
  root_dir = os.path.normpath(
      os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

  mlperf_log.setdefault(
      benchmark="resnet", root_dir=root_dir, extra_print=True, prefix="TEST")
  mlperf_log.mlperf_print(constants.RUN_START)
  mlperf_log.mlperf_print(constants.EVAL_ACCURACY, value=0.99)
  mlperf_log.mlperf_print(
          constants.EPOCH_STOP, metadata={"first_epoch_num": 1})

if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "manual":
    manual_test()
  else:
    unittest.main()
