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
"""Useful functions for the tests in this directory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import itertools



import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow as tf


def random_binary_sparse_matrix(num_non_zeros, num_rows, num_cols):
  values = np.ones(num_non_zeros)
  indices = list(itertools.product(range(num_rows), range(num_cols)))
  sampled = np.random.choice(np.arange(num_rows * num_cols), num_non_zeros,
                             replace=False)
  sampled_indices = [indices[i] for i in sampled]
  rows = [i for (i, _) in sampled_indices]
  cols = [j for (_, j) in sampled_indices]
  return sparse.csr_matrix((values, (rows, cols)), shape=[num_rows, num_cols])


def sparse_to_df(sparse_matrix):
  sparse_matrix_as_coo = sparse_matrix.tocoo()
  row_indices = sparse_matrix_as_coo.row
  col_indices = sparse_matrix_as_coo.col
  data = sparse_matrix_as_coo.data

  return pd.DataFrame({"row": row_indices, "col": col_indices, "data": data})


def all_close(left_nd_array, right_nd_array, tol=1e-6):
  return np.abs(left_nd_array - right_nd_array).max() <= tol


def read_from_serialized_file(file_path):
  with tf.gfile.Open(file_path, "rb") as infile:
    return pickle.load(infile)


def read_all_from_serialized_file(file_path):
  objects = []
  with tf.gfile.Open(file_path, "rb") as infile:
    while True:
      try:
        objects.append(pickle.load(infile))
      except EOFError:
        break
  return objects
