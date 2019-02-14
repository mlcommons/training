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
"""Tests for util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf

import util
from test_util import random_binary_sparse_matrix
from test_util import sparse_to_df


class UtilTest(tf.test.TestCase):

  def test_convert_df_to_sparse_matrix(self):
    np.random.seed(0)

    num_rows = 16
    num_cols = 32
    num_non_zeros = 40

    sparse_matrix = random_binary_sparse_matrix(
        num_non_zeros, num_rows, num_cols)

    sparse_matrix_as_df = sparse_to_df(sparse_matrix)
    sparse_matrix_as_df["data"] = 1.0

    sparse_matrix_as_df_as_sparse = util.convert_df_to_sparse_matrix(
        sparse_matrix_as_df, shape=(num_rows, num_cols))

    self.assertEqual(sparse_matrix.shape, sparse_matrix_as_df_as_sparse.shape)

    # Check that there are no non-zeros in the inequality boolean matrix.
    self.assertEqual(
        (sparse_matrix != sparse_matrix_as_df_as_sparse).nnz, 0)


if __name__ == "__main__":
  tf.test.main()
