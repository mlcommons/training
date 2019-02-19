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
"""Tests for graph_reduction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf

import graph_reduction
from test_util import all_close


class GraphReductionTest(tf.test.TestCase):

  def test_closest_column_orthogonal_matrix_is_column_orthogonal(self):
    np.random.seed(0)

    num_rows = 16
    num_cols = 4

    matrix = np.random.uniform(size=(num_rows, num_cols))

    col_orth_matrix = graph_reduction._closest_column_orthogonal_matrix(matrix)

    self.assertTrue(all_close(
        np.matmul(col_orth_matrix.T, col_orth_matrix), np.eye(num_cols)))

  def test_reduced_matrix_has_right_size(self):
    np.random.seed(0)

    num_rows = 64
    num_cols = 16

    reduced_num_rows = 12
    reduced_num_cols = 3

    svd_size = 8

    u = np.random.uniform(size=(num_rows, svd_size))
    v = np.random.uniform(size=(svd_size, num_cols))
    s = np.random.uniform(size=svd_size)

    resized_matrix = graph_reduction.resize_matrix(
        (u, s, v), reduced_num_rows, reduced_num_cols)

    self.assertEqual(resized_matrix.shape[0], reduced_num_rows)
    self.assertEqual(resized_matrix.shape[1], reduced_num_cols)

  def test_reduced_matrix_has_same_singular_value_spectrum(self):
    np.random.seed(0)

    num_rows = 64
    num_cols = 16

    reduced_num_rows = 12
    reduced_num_cols = 3

    svd_size = 8

    u = np.random.uniform(size=(num_rows, svd_size))
    v = np.random.uniform(size=(svd_size, num_cols))
    s = sorted(np.random.uniform(size=svd_size))

    resized_matrix = graph_reduction.resize_matrix(
        (u, s, v), reduced_num_rows, reduced_num_cols)

    resized_matrix_s = np.linalg.svd(resized_matrix, compute_uv=False)

    self.assertTrue(all_close(s[::-1][:reduced_num_cols], resized_matrix_s))

  def test_normalize_matrix(self):
    np.random.seed(0)

    matrix = np.random.normal(size=(16, 32))
    normalized_matrix = graph_reduction.normalize_matrix(matrix)

    self.assertEqual(matrix.shape, normalized_matrix.shape)

    self.assertGreaterEqual(normalized_matrix.min(), 0.0)
    self.assertLessEqual(normalized_matrix.max(), 1.0)


if __name__ == "__main__":
  tf.test.main()
