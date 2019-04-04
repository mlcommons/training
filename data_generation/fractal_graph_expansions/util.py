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
"""Miscellaneous utilities for fractal_graph_expansions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import cPickle as pickle



from absl import logging

import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow as tf


def load_df_from_file(file_path, sep=",", header=0):
  """Wrapper around pandas' read_csv."""
  with tf.gfile.Open(file_path) as infile:
    df = pd.read_csv(infile, sep=sep, header=header)
  return df


def convert_df_to_sparse_matrix(df, shape=None, row_name="row", col_name="col",
                                data_name="data"):
  row = df[row_name].values
  col = df[col_name].values
  data = df[data_name].values

  return sparse.csr_matrix((data, (row, col)), shape=shape)


def describe_rating_df(df, df_name=""):
  num_ratings = len(df)
  num_users = len(set(df["row"].values))
  num_items = len(set(df["col"].values))

  logging.info("%d users in ratings dataframe %s", num_users, df_name)
  logging.info("%d items in ratings dataframe %s", num_items, df_name)
  logging.info("%d ratings in ratings datagrame %s", num_ratings, df_name)

  return num_users, num_items, num_ratings


def serialize_to_file(obj, file_name, append=False):
  """Pickle obj to file_name."""
  logging.info("Serializing to file %s.", file_name)
  with tf.gfile.Open(file_name, "a+" if append else "wb") as output_file:
    pickle.dump(obj, output_file)
  logging.info("Done serializing to file %s.", file_name)

def savez_two_column(matrix, row_offset, file_name, append=False):
  """Savez_compressed obj to file_name."""
  logging.info("Saving obj to file in two column .npz format %s.", file_name)
  tc = []
  for u, items in enumerate(matrix):
    user = row_offset + u
    for item in items:
      tc.append([user, item])
  
  np.savez_compressed(file_name, np.asarray(tc))
  logging.info("Done saving to file %s.", file_name)

def sorted_product_set(array_a, array_b):
  """Compute the product set of array_a and array_b and sort it."""
  return np.sort(
      np.concatenate(
          [array_a[i] * array_b for i in xrange(len(array_a))], axis=0)
  )[::-1]


def write_metadata_to_file(metadata_named_tuple, metadata_out_path, tag=""):
  logging.info("Writing %s metadata file to %s", tag, metadata_out_path)
  serialize_to_file(metadata_named_tuple, file_name=metadata_out_path)
  logging.info("Done writing %s metadata file to %s", tag, metadata_out_path)


def sparse_where_equal(coo_matrix, target_value):

  cond_is_true = coo_matrix.data == target_value
  data_where = coo_matrix.data[cond_is_true]
  row_where = coo_matrix.row[cond_is_true]
  col_where = coo_matrix.col[cond_is_true]

  return sparse.csr_matrix(
      (data_where, (row_where, col_where)), shape=coo_matrix.shape)
