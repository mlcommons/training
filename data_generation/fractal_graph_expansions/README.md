# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

This is a directory used to produce larger versions of MovieLens through
  fractal expansions. The expansion is stochastic, see equation (2) and
  algorithm (1) in https://arxiv.org/abs/1901.08910 for more details
  (temporary, to be replaced by white paper). The ratings are binarized,
  each rating becoming 1.

If you use this data please cite:
  * https://dl.acm.org/citation.cfm?doid=2866565.2827872
  * https://arxiv.org/abs/1901.08910 (temporary, to be replaced by white paper)

Dependencies:
  * numpy (tested with version 1.16.1)
  * absl-py (tested with version 0.7.0)
  * pandas (tested with version 0.24.1)
  * scipy (tested with version 1.2.1)
  * tensorflow (tested with version 1.12.0)
  * scikit-image (tested with version 0.14.2)
  * sklearn (tested with version 0.20.2)

How to run (takes ~30 mins on a recent desktop):
  1) Download MovieLens20m from https://grouplens.org/datasets/movielens/
  (permalink http://grouplens.org/datasets/movielens/20m/).
  2) python run_expansion.py will generate the data set with the following flags
    a) --input_csv_file, the path to the ratings.csv file downloaded
      from MovieLens.
    b) --num_row_multiplier, the multiplier for the number of users.
    c) --num_col_multiplier, the multiplier for the number of items.
    d) --output_prefix, the path to the output files including their prefix.

A train and test set will be generated. No information from the test set is
  available in the train set.
The train set data will consist of num_row_multiplier shards named
  output_prefix_train.pkl_%d % shard for shard in range(num_row_multiplier).
The test set data will consist of num_row_multiplier shards named
  output_prefix_test.pkl_%d % shard for shard in range(num_row_multiplier).
Each shard is a pickled list of numpy arrays, each array corresponds to an user
  and entails the sequence of item indices corresponding to the items the user
  has interacted with.
The train and test sets will also each feature a separate metadata file
  output_prefix_train/test_metadata.pkl. The metadata contains a pickled
  graph_expansion.SparseMatrixMetadata object entailing the number of
  interactions, users and items in each data set.

If the original rating matrix (after filtering) has (n, m) (users, items) then
  the synthesized matrix will have about
  (n x num_row_multiplier, m x num_col_multiplier) (users, items).

Actual users with less than two distinct rating timestamps are dropped from
  the original data set. Synthetic users with no ratings in either the synthetic
  train or test set are all dropped. Items without ratings may be present in
  the train and/or test sets.

Other useful flags:
  1) --min_dropout_rate, decreasing/increasing this value will result in
    a denser/sparser generated data set.
  2) --max_dropout_rate, decreasing/increasing this value will result in
    a denser/sparser generated data set.

