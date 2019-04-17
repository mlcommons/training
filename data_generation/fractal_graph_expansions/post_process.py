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
"""Post process expanded data set to output to user/item/rating csv files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import pickle
import pandas as pd
import tensorflow as tf


flags.DEFINE_string("output_prefix",
                    "",
                    "Prefix to the path of the pickle files that have been "
                    "produced. output_prefix_train.csv and "
                    "output_prefix_test.csv will be created.")
flags.DEFINE_integer("num_shards",
                     16,
                     "Number of shards used to output data.")

FLAGS = flags.FLAGS


def _read_from_serialized_file(file_name):
  with tf.gfile.Open(file_name, "rb") as infile:
    return pickle.load(infile)


def _convert_pickled_shards_to_csv(prefix):
  logging.info("Processing %d shards with prefix %s.", FLAGS.num_shards, prefix)
  for shard_idx in range(FLAGS.num_shards):
    shard = _read_from_serialized_file(prefix + ".pkl_%d" % shard_idx)
    logging.info("Converting shard %d to csv format.", shard_idx)
    with tf.gfile.Open(prefix + ".csv_%d" % shard_idx, "wb") as outfile:
      for user_idx, item_array in enumerate(shard):
        for item_idx in item_array:
          outfile.write("%d,%d,1\n" % (user_idx, item_idx))
    logging.info("Done converting shard %d to csv format.", shard_idx)
  logging.info("Done processing %d shards with prefix %s.", FLAGS.num_shards, prefix)


def main(_):
  
   _convert_pickled_shards_to_csv(FLAGS.output_prefix + "_train")
   _convert_pickled_shards_to_csv(FLAGS.output_prefix + "_test")


if __name__ == "__main__":
  app.run(main)
