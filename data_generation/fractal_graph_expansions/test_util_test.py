# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for test_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

import test_util


class TestUtilTest(tf.test.TestCase):

  def test_random_binary_sparse_matrix(self):
    num_rows = 16
    num_cols = 32
    num_non_zeros = 100

    sparse_matrix = test_util.random_binary_sparse_matrix(
        num_non_zeros, num_rows, num_cols)

    self.assertEqual(sparse_matrix.shape[0], num_rows)
    self.assertEqual(sparse_matrix.shape[1], num_cols)
    self.assertEqual(sparse_matrix.nnz, num_non_zeros)


if __name__ == "__main__":
  tf.test.main()
