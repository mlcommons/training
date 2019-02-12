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
"""Fast shuffle and dropout of large sparse matrices.

For speed matrices are encoded in a pandas dataframe. To make the behavior
of the following operators deterministic, it is sufficient to setup numpy's
random seed before these operators are called (numpy.random.seed(seed_value)).
Note also that callers running the functions below in parallel are responsible
for guaranteeing that the corresponding underlying sequences of random numbers
(which will be genereted in parallel) are non overlapping.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl import flags
from absl import logging

import numpy as np

from scipy import sparse
from sklearn.utils import shuffle

flags.DEFINE_float("min_dropout_rate",
                   0.05,
                   "Mininum dropout rate in shuffle_sparse_matrix if none is "
                   "specified. A lower dropout rate will be clipped to "
                   "the minimum value.")
flags.DEFINE_float("max_dropout_rate",
                   0.99,
                   "Maximum dropout rate in shuffle_sparse_matrix if none is "
                   "specified. A greater dropout rate will be clipped to "
                   "the maximum value.")

FLAGS = flags.FLAGS


def _dropout_sparse_coo_matrix(sparse_matrix, rate,
                               min_dropout_rate, max_dropout_rate):
  """Drop values from a sparse matrix encoded as a SciPy coo matrix.

  Args:
    sparse_matrix: a SciPy coo sparse matrix.
    rate: if rate > 0 then non-zero elements of the input matrix
      will be droped uniformly at random.
    min_dropout_rate: minimum value for the dropout rate. If None
      FLAGS.min_dropout_rate is used. If dropout_rate is lower than
      min_dropout_rate it will clipped to min_dropout_rate.
    max_dropout_rate: minimum value for the dropout rate. If None
      FLAGS.max_dropout_rate is used. If dropout_rate is greater than
      max_dropout_rate it will clipped to max_dropout_rate.

  Returns:
    A SciPy coo matrix containing those non zero elements that have not been
    dropped out.
  """
  if min_dropout_rate is None:
    min_dropout_rate = FLAGS.min_dropout_rate

  if max_dropout_rate is None:
    max_dropout_rate = FLAGS.max_dropout_rate

  if min_dropout_rate > max_dropout_rate:
    raise ValueError("min_dropout_rate (%f) should be less or equal to "
                     "max_dropout_rate (%f)"
                     % (min_dropout_rate, max_dropout_rate))

  max_frac = 1.0 - min_dropout_rate
  min_frac = 1.0 - max_dropout_rate
  sampling_rate = 1.0 - rate

  sampled_fraction = min(max(sampling_rate, min_frac), max_frac)
  if sampled_fraction != sampling_rate:
    logging.warning("Minimum sampling rate is %2f.", min_frac)
    logging.warning("Maximum sampling rate is %2f.", max_frac)
    logging.warning("Desired sampling rate is %2f.", sampling_rate)
    logging.warning("Desired sampling rate %2f clipped to %2f.", sampling_rate,
                    sampled_fraction)

  num_sampled = min(
      max(int(sparse_matrix.nnz * sampled_fraction), 1), sparse_matrix.nnz)
  sampled_indices = np.random.choice(sparse_matrix.nnz, size=num_sampled,
                                     replace=False)

  return sparse.coo_matrix((sparse_matrix.data[sampled_indices],
                            (sparse_matrix.row[sampled_indices],
                             sparse_matrix.col[sampled_indices])),
                           shape=sparse_matrix.shape)


def shuffle_sparse_coo_matrix(sparse_matrix, dropout_rate=0.0,
                              min_dropout_rate=None, max_dropout_rate=None):
  """Shuffle sparse matrix encoded as a SciPy coo matrix.

  Args:
    sparse_matrix: a SciPy coo sparse matrix.
    dropout_rate: if dropout_rate > 0 then non-zero elements of the input matrix
      will be droped uniformly at random.
    min_dropout_rate: minimum value for the dropout rate. If None
      FLAGS.min_dropout_rate is used.
    max_dropout_rate: minimum value for the dropout rate. If None
      FLAGS.max_dropout_rate is used.

  Returns:
    A SciPy csr_matrix entailing the randomized interactions.
  """

  if (dropout_rate < 0.0) or (dropout_rate >= 1.0):
    raise ValueError("Dropout rate should be in [0, 1) but is %f"
                     % dropout_rate)

  (num_rows, num_cols) = sparse_matrix.shape
  shuffled_rows = shuffle(np.arange(num_rows))
  shuffled_cols = shuffle(np.arange(num_cols))

  if dropout_rate > 0.0:
    sparse_matrix = _dropout_sparse_coo_matrix(
        sparse_matrix, dropout_rate, min_dropout_rate, max_dropout_rate)

  new_row = np.take(shuffled_rows, sparse_matrix.row)
  new_col = np.take(shuffled_cols, sparse_matrix.col)

  return sparse.csr_matrix(
      (sparse_matrix.data, (new_row, new_col)), shape=(num_rows, num_cols))
