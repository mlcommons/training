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
"""Tests for graph_analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf

import graph_analysis
from test_util import all_close
from test_util import random_binary_sparse_matrix


class GraphAnalysisTest(tf.test.TestCase):

  def test_svd_is_valid(self):
    np.random.seed(0)

    num_rows = 16
    num_cols = 32
    num_non_zeros = 100
    input_matrix = random_binary_sparse_matrix(
        num_non_zeros, num_rows, num_cols)
    k = 5

    (u, s, v) = graph_analysis.sparse_svd(input_matrix, k, max_iter=16)

    # Check that singular values are in increasing order:
    for i in range(k - 1):
      self.assertGreaterEqual(s[i + 1], s[i])

    # Check that singular vector matrices have the right shapes.
    self.assertEqual(u.shape[0], num_rows)
    self.assertEqual(u.shape[1], k)

    self.assertEqual(v.shape[0], k)
    self.assertEqual(v.shape[1], num_cols)

    # Check that u is column orthogonal.
    self.assertTrue(all_close(np.matmul(u.T, u), np.eye(k)))

    # Check that v is row orthogonal.
    self.assertTrue(all_close(np.matmul(v, v.T), np.eye(k)))


if __name__ == "__main__":
  tf.test.main()
