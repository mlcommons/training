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
"""Tests for random_matrix_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections



import numpy as np
import tensorflow as tf

import random_matrix_ops
from test_util import random_binary_sparse_matrix


def _unique_value_counts(array):
  return sorted(collections.Counter(array).values())


class RandomMatrixOpsTest(tf.test.TestCase):

  def test_dropout_coo(self):
    np.random.seed(0)

    dropout_rate = 0.5

    num_rows = 16
    num_cols = 32
    num_non_zeros = 20

    sparse_matrix = random_binary_sparse_matrix(num_non_zeros, num_rows,
                                                num_cols)

    sparse_coo_matrix = sparse_matrix.tocoo()

    expected_num_sampled = int(num_non_zeros * dropout_rate)
    sampled_coo_matrix = random_matrix_ops._dropout_sparse_coo_matrix(
        sparse_coo_matrix,
        dropout_rate,
        min_dropout_rate=0.0,
        max_dropout_rate=1.0)

    self.assertEqual(sampled_coo_matrix.nnz, expected_num_sampled)

  def test_shuffle_sparse_coo_matrix_preserves_unique_counts(self):
    np.random.seed(0)

    num_rows = 16
    num_cols = 8
    num_non_zeros = 20

    sparse_matrix = random_binary_sparse_matrix(num_non_zeros, num_rows,
                                                num_cols)

    sparse_coo_matrix = sparse_matrix.tocoo()

    # No values should be dropped out.
    shuffled_sparse_coo_matrix = random_matrix_ops.shuffle_sparse_coo_matrix(
        sparse_coo_matrix,
        dropout_rate=0.0,
        min_dropout_rate=0.0,
        max_dropout_rate=1.0).tocoo()

    original_sorted_row_counts = _unique_value_counts(sparse_coo_matrix.row)
    original_sorted_col_counts = _unique_value_counts(sparse_coo_matrix.col)

    shuffled_sorted_row_counts = _unique_value_counts(
        shuffled_sparse_coo_matrix.row)
    shuffled_sorted_col_counts = _unique_value_counts(
        shuffled_sparse_coo_matrix.col)

    self.assertEqual(
        original_sorted_row_counts, shuffled_sorted_row_counts)
    self.assertEqual(
        original_sorted_col_counts, shuffled_sorted_col_counts)


if __name__ == "__main__":
  tf.test.main()
