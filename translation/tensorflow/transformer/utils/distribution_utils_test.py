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
""" Tests for distribution util functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import distribution_utils

mock = tf.compat.v1.test.mock


class GetNumGPUsTest(tf.test.TestCase):
  """Tests for get_num_gpus."""
  def test_zero_gpu(self):
    num = distribution_utils.get_num_gpus(num_gpus=0)
    self.assertEqual(num, 0)

  def test_one_gpu(self):
    num = distribution_utils.get_num_gpus(num_gpus=1)
    self.assertEqual(num, 1)

  def test_multi_gpu(self):
    num = distribution_utils.get_num_gpus(num_gpus=5)
    self.assertEqual(num, 5)

  @mock.patch("tensorflow.test.is_gpu_available", return_value=True)
  def test_default_available_gpu(self, mock_is_gpu_available):
    num = distribution_utils.get_num_gpus(num_gpus=None)
    self.assertEqual(num, 1)

  @mock.patch("tensorflow.test.is_gpu_available", return_value=False)
  def test_default_unavailable_gpu(self, mock_is_gpu_available):
    num = distribution_utils.get_num_gpus(num_gpus=None)
    self.assertEqual(num, 0)


class GetDistributionStrategyTest(tf.test.TestCase):
  """Tests for get_distribution_strategy."""
  def test_one_device_strategy_cpu(self):
    ds = distribution_utils.get_distribution_strategy(
        distribution_strategy="one_device", num_gpus=0)
    self.assertEqual(ds.num_replicas_in_sync, 1)
    self.assertEqual(len(ds.extended.worker_devices), 1)
    self.assertIn('CPU', ds.extended.worker_devices[0])

  def test_one_device_strategy_gpu(self):
    ds = distribution_utils.get_distribution_strategy(
        distribution_strategy="one_device", num_gpus=1)
    self.assertEqual(ds.num_replicas_in_sync, 1)
    self.assertEqual(len(ds.extended.worker_devices), 1)
    self.assertIn('GPU', ds.extended.worker_devices[0])

  def test_mirrored_strategy(self):
    ds = distribution_utils.get_distribution_strategy(
        distribution_strategy="mirrored", num_gpus=5)
    self.assertEqual(ds.num_replicas_in_sync, 5)
    self.assertEqual(len(ds.extended.worker_devices), 5)
    for device in ds.extended.worker_devices:
      self.assertIn('GPU', device)


if __name__ == "__main__":
  tf.test.main()
