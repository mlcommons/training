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
"""Fractal graph expander.

Detailed analysis in the deterministic case provided in
https://arxiv.org/abs/1901.08910.
Please refer the paper if you use this code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from six.moves import xrange

from absl import logging

from scipy import sparse

import util
from random_matrix_ops import shuffle_sparse_coo_matrix


SparseMatrixMetadata = collections.namedtuple(
    "SparseMatrixMetadata", [
        "num_interactions",
        "num_rows",
        "num_cols",
    ])


def _compute_and_write_row_block(
    i, left_matrix, right_matrix, train_indices_out_path, test_indices_out_path,
    remove_empty_rows):
  """Compute row block (shard) of expansion for row i of the left_matrix.

  Compute a shard of the randomized Kronecker product and dump it on the fly.
  A standard Kronecker product between matrices A and B produces
                        [[a_11 B, ..., a_1n B],
                                  ...
                         [a_m1 B, ..., a_mn B]]
    (if A's size is (m, n) and B's size is (p, q) then A Kronecker B has size
    (m p, n q)).
    Here we modify the standard Kronecker product expanding matrices in
    https://cs.stanford.edu/~jure/pubs/kronecker-jmlr10.pdf
    and randomize each block-wise operation a_ij B in the Kronecker product as
    in https://arxiv.org/pdf/1901.08910.pdf section III.4.
    The matrix we produce is
                       [[F(a_11, B, w_11), ..., F(a_1n, B, w_1n)],
                                           ...
                       [F(a_m1, B, w_m1), ... , F(a_mn, B, w_mn)]]
    where (w_ij) is a sequence of pseudo random numbers and F is randomized
    operator which will:
      1) Shuffle rows and columns of B independently at random;
      2) Dropout elements of B with a rate 1 - a_ij to compute
        F(a_ij, B, w_ij).
    (It is noteworthy that there is an abuse of notation above when writing
    F(a_ij, B, w_ij) as each block-wise operation will in fact consume
    multiple elements of the sequence (w_ij)).
  Each shard of index i consists of [F(a_i1, B, w_i1), ..., F(a_in, B, w_in)]

  Args:
    i: index of the shard. The rows i * m to (i + 1) * m of the full synthetic
      matrix matrix will be computed and dumpted to file.
    left_matrix: sparse SciPy csr matrix with values in [0, 1].
    right_matrix: sparse SciPy coo signed binary matrix. +1 values correspond
      to train set and -1 values correspond to test set.
    train_indices_out_path: path to output train file. The non zero indices of
      the resulting sparse matrix are dumped as a series of pickled records.
      '_i' will be used as a suffix for the shard's output file. The shard
      contains a pickled list of list each of which corresponds to a users.
    test_indices_out_path: path to output train file. The non zero indices of
      the resulting sparse matrix are dumped as a series of pickled records.
     '_i' will be used as a suffix for the shard's output file. The shard
      contains a pickled list of list each of which corresponds to a users.
    remove_empty_rows: whether to remove rows from the synthetic train and
      test matrices which are not present in the train or the test matrix.

  Returns:
    (num_removed_rows, metadata, train_metadata, test_metadata): an integer
      specifying the number of rows dropped because of dropout followed by
      a triplet of SparseMatrixMetadata corresponding to the overall shard,
      train shard and test shard.
  """

  kron_blocks = []

  num_rows = 0
  num_removed_rows = 0
  num_interactions = 0
  num_train_interactions = 0
  num_test_interactions = 0

  # Construct blocks
  for j in xrange(left_matrix.shape[1]):

    dropout_rate = 1.0 - left_matrix[i, j]
    kron_block = shuffle_sparse_coo_matrix(right_matrix, dropout_rate)

    if not set(kron_block.data).issubset({1, -1}):
      raise ValueError("Values of sparse matrix should be -1 or 1 but are: ",
                       set(kron_block.data))

    kron_blocks.append(kron_block)

    logging.info("Done with element (%d, %d)", i, j)

  rows_to_write = sparse.hstack(kron_blocks).tocoo()

  train_rows_to_write = util.sparse_where_equal(rows_to_write, 1)
  test_rows_to_write = util.sparse_where_equal(rows_to_write, -1)

  logging.info("Producing data set row by row")

  all_train_items_to_write = []
  all_test_items_to_write = []
  # Write Kronecker product line per line.
  for k in xrange(right_matrix.shape[0]):

    train_items_to_write = train_rows_to_write.getrow(k).indices
    test_items_to_write = test_rows_to_write.getrow(k).indices

    # for users with > 1 test items, keep only the first one
    if len(test_items_to_write) > 1:
        test_items_to_write = test_items_to_write[:1]

    num_train = train_items_to_write.shape[0]
    num_test = test_items_to_write.shape[0]

    if remove_empty_rows and ((not num_train) or (not num_test)):
      logging.info("Removed empty output row %d.",
                   i * left_matrix.shape[0] + k)
      num_removed_rows += 1
      continue

    num_rows += 1
    num_interactions += num_train + num_test
    num_train_interactions += num_train
    num_test_interactions += num_test

    all_train_items_to_write.append(train_items_to_write)
    all_test_items_to_write.append(test_items_to_write)

    if k % 1000 == 0:
      logging.info("Done producing data set row %d.", k)

  logging.info("Done producing data set row by row.")

  util.savez_two_column(
      all_train_items_to_write, 
      row_offset=(i * right_matrix.shape[0]),
      file_name=train_indices_out_path + ("_%d" % i))
  util.savez_two_column(
      all_test_items_to_write, 
      row_offset=(i * right_matrix.shape[0]),
      file_name=test_indices_out_path + ("_%d" % i))

  num_cols = rows_to_write.shape[1]
  metadata = SparseMatrixMetadata(num_interactions=num_interactions,
                                  num_rows=num_rows, num_cols=num_cols)
  train_metadata = SparseMatrixMetadata(num_interactions=num_train_interactions,
                                        num_rows=num_rows, num_cols=num_cols)
  test_metadata = SparseMatrixMetadata(num_interactions=num_test_interactions,
                                       num_rows=num_rows, num_cols=num_cols)

  logging.info("Done with left matrix row %d.", i)
  logging.info("%d interactions written in shard.", num_interactions)
  logging.info("%d rows removed in shard.", num_removed_rows)
  logging.info("%d train interactions written in shard.",
               num_train_interactions)
  logging.info("%d test interactions written in shard.",
               num_test_interactions)

  return (num_removed_rows, metadata, train_metadata, test_metadata)


def output_randomized_kronecker_to_pickle(
    left_matrix, right_matrix,
    train_indices_out_path, test_indices_out_path,
    train_metadata_out_path=None, test_metadata_out_path=None,
    remove_empty_rows=True):
  """Compute randomized Kronecker product and dump it on the fly.

  A standard Kronecker product between matrices A and B produces
                        [[a_11 B, ..., a_1n B],
                                  ...
                         [a_m1 B, ..., a_mn B]]
    (if A's size is (m, n) and B's size is (p, q) then A Kronecker B has size
    (m p, n q)).
    Here we modify the standard Kronecker product expanding matrices in
    https://cs.stanford.edu/~jure/pubs/kronecker-jmlr10.pdf
    and randomize each block-wise operation a_ij B in the Kronecker product as
    in https://arxiv.org/pdf/1901.08910.pdf section III.4.
    The matrix we produce is
                       [[F(a_11, B, w_11), ..., F(a_1n, B, w_1n)],
                                           ...
                       [F(a_m1, B, w_m1), ... , F(a_mn, B, w_mn)]]
    where (w_ij) is a sequence of pseudo random numbers and F is randomized
    operator which will:
      1) Shuffle rows and columns of B independently at random;
      2) Dropout elements of B with a rate 1 - a_ij to compute
        F(a_ij, B, w_ij).
    (It is noteworthy that there is an abuse of notation above when writing
    F(a_ij, B, w_ij) as each block-wise operation will in fact consume
    multiple elements of the sequence (w_ij)).

  Args:
    left_matrix: sparse SciPy csr matrix with values in [0, 1].
    right_matrix: sparse SciPy coo signed binary matrix. +1 values correspond
      to train set and -1 values correspond to test set.
    train_indices_out_path: path to output train file. The non zero indices of
      the resulting sparse matrix are dumped as a series of pickled records.
      As many shard will be created as there are rows in left matrix. The shard
      corresponding to row i in the left matrix has the suffix _i appended to
      its file name. Each shard contains a pickled list of list each of which
      corresponds to a users.
    test_indices_out_path: path to output train file. The non zero indices of
      the resulting sparse matrix are dumped as a series of pickled records.
      As many shard will be created as there are rows in left matrix. The shard
      corresponding to row i in the left matrix has the suffix _i appended to
      its file name. Each shard contains a pickled list of list each of which
      corresponds to a users.
    train_metadata_out_path: path to optional complementary output file
      containing the number of train rows (r), columns (c) and non zeros (nnz)
      in a pickled SparseMatrixMetadata named tuple.
    test_metadata_out_path: path to optional complementary output file
      containing the number of test rows (r), columns (c) and non zeros (nnz)
      in a pickled SparseMatrixMetadata named tuple.
    remove_empty_rows: whether to remove rows from the synthetic train and
      test matrices which are not present in the train or the test matrix.

  Returns:
    (metadata, train_metadata, test_metadata) triplet of SparseMatrixMetadata
      corresponding to the overall data set, train data set and test data set.
  """
  logging.info("Writing item sequences to pickle files %s and %s.",
               train_indices_out_path, test_indices_out_path)

  num_rows = 0
  num_removed_rows = 0
  num_cols = left_matrix.shape[1] * right_matrix.shape[1]
  num_interactions = 0

  num_train_interactions = 0
  num_test_interactions = 0

  if not set(right_matrix.data).issubset({-1, 1}):
    raise ValueError(
        "Values of sparse matrix should be -1 or 1 but are:",
        set(right_matrix.data))

  for i in xrange(left_matrix.shape[0]):

    (shard_num_removed_rows, shard_metadata, shard_train_metadata,
     shard_test_metadata) = _compute_and_write_row_block(
         i, left_matrix, right_matrix, train_indices_out_path,
         test_indices_out_path, remove_empty_rows)

    num_rows += shard_metadata.num_rows
    num_removed_rows += shard_num_removed_rows
    num_interactions += shard_metadata.num_interactions
    num_train_interactions += shard_train_metadata.num_interactions
    num_test_interactions += shard_test_metadata.num_interactions

    logging.info("%d total interactions written.", num_interactions)
    logging.info("%d total rows removed.", num_removed_rows)
    logging.info("%d total train interactions written.", num_train_interactions)
    logging.info("%d toal test interactions written.", num_test_interactions)

  logging.info("Done writing.")

  metadata = SparseMatrixMetadata(
      num_interactions=num_interactions,
      num_rows=num_rows, num_cols=num_cols)
  train_metadata = SparseMatrixMetadata(
      num_interactions=num_train_interactions,
      num_rows=num_rows, num_cols=num_cols)
  test_metadata = SparseMatrixMetadata(
      num_interactions=num_test_interactions,
      num_rows=num_rows, num_cols=num_cols)

  if train_metadata_out_path is not None:
    util.write_metadata_to_file(
        train_metadata, train_metadata_out_path, tag="train")
  if test_metadata_out_path is not None:
    util.write_metadata_to_file(
        test_metadata, test_metadata_out_path, tag="test")

  return metadata, train_metadata, test_metadata
