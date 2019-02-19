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
"""Tests for graph_expansion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools



import numpy as np
import tensorflow as tf

import graph_expansion
from test_util import random_binary_sparse_matrix
from test_util import read_from_serialized_file


class GraphExpansionTest(tf.test.TestCase):

  def test_produces_synthetic_interactions_with_right_shape(self):
    np.random.seed(0)

    left_matrix_num_rows = 4
    left_matrix_num_cols = 8
    left_matrix = np.ones((left_matrix_num_rows, left_matrix_num_cols))

    right_matrix_num_rows = 16
    right_matrix_num_cols = 32
    right_matrix = random_binary_sparse_matrix(
        50, right_matrix_num_rows,
        right_matrix_num_cols) - random_binary_sparse_matrix(
            50, right_matrix_num_rows, right_matrix_num_cols)
    right_matrix = right_matrix.tocoo()
    right_matrix_num_non_zeros = right_matrix.nnz

    right_matrix_num_train = (right_matrix == 1).nnz
    right_matrix_num_test = (right_matrix == -1).nnz

    train_output_file = self.create_tempfile("temp_train.pkl")
    test_output_file = self.create_tempfile("temp_test.pkl")

    train_meta_output_file = self.create_tempfile("temp_train_meta.pkl")
    test_meta_output_file = self.create_tempfile("temp_test_meta.pkl")

    (metadata, train_metadata,
     test_metadata) = graph_expansion.output_randomized_kronecker_to_pickle(
         left_matrix, right_matrix,
         train_output_file.full_path, test_output_file.full_path,
         train_meta_output_file.full_path, test_meta_output_file.full_path,
         remove_empty_rows=False)

    # Left matrix is filled with 1s here.
    self.assertEqual(metadata.num_interactions,
                     left_matrix_num_rows * left_matrix_num_cols *
                     right_matrix_num_non_zeros)
    self.assertEqual(metadata.num_rows,
                     left_matrix_num_rows * right_matrix_num_rows)
    self.assertEqual(metadata.num_cols,
                     left_matrix_num_cols * right_matrix_num_cols)

    # Right matrix is filled with 1s here so there should be no test set.
    self.assertEqual(train_metadata.num_interactions,
                     left_matrix_num_rows * left_matrix_num_cols *
                     right_matrix_num_train)
    self.assertEqual(train_metadata.num_rows,
                     left_matrix_num_rows * right_matrix_num_rows)
    self.assertEqual(train_metadata.num_cols,
                     left_matrix_num_cols * right_matrix_num_cols)

    self.assertEqual(test_metadata.num_interactions,
                     left_matrix_num_rows * left_matrix_num_cols *
                     right_matrix_num_test)
    self.assertEqual(test_metadata.num_rows,
                     left_matrix_num_rows * right_matrix_num_rows)
    self.assertEqual(test_metadata.num_cols,
                     left_matrix_num_cols * right_matrix_num_cols)

    pickled_train_metadata = read_from_serialized_file(
        train_meta_output_file.full_path)
    pickled_test_metadata = read_from_serialized_file(
        test_meta_output_file.full_path)

    self.assertEqual(train_metadata, pickled_train_metadata)
    self.assertEqual(test_metadata, pickled_test_metadata)

  def test_produces_synthetic_interactions_with_right_content(self):
    np.random.seed(0)

    left_matrix_num_rows = 4
    left_matrix_num_cols = 8
    left_matrix = np.ones((left_matrix_num_rows, left_matrix_num_cols))

    right_matrix_num_rows = 16
    right_matrix_num_cols = 32
    right_matrix_num_non_zeros = 100
    right_matrix = random_binary_sparse_matrix(
        right_matrix_num_non_zeros, right_matrix_num_rows,
        right_matrix_num_cols)
    right_matrix = right_matrix.tocoo()

    num_shards = len(left_matrix)
    train_output_file = self.create_tempfile("temp_train.pkl")
    train_output_shards = [
        self.create_tempfile("temp_train.pkl_%d" % shard_idx)
        for shard_idx in range(num_shards)]
    test_output_file = self.create_tempfile("temp_test.pkl")

    train_meta_output_file = self.create_tempfile("temp_train_meta.pkl")
    test_meta_output_file = self.create_tempfile("temp_test_meta.pkl")

    graph_expansion.output_randomized_kronecker_to_pickle(
        left_matrix, right_matrix,
        train_output_file.full_path, test_output_file.full_path,
        train_meta_output_file.full_path, test_meta_output_file.full_path,
        remove_empty_rows=False)

    serialized_rows = []
    for shard_output_file in train_output_shards:
      serialized_rows.extend(read_from_serialized_file(
          shard_output_file.full_path))

    self.assertLen(serialized_rows,
                   left_matrix_num_rows * right_matrix_num_rows)

    output_item_set = set(itertools.chain(*serialized_rows))
    self.assertEqual(output_item_set,
                     set(range(left_matrix_num_cols * right_matrix_num_cols)))


if __name__ == "__main__":
  tf.test.main()
