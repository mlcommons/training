# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Utilities to create, read, write tf.Examples.'''
import functools
import numpy as np
# import tensorflow as tf
import random

import torch
import torch.utils.data as Data

import coords
import features as features_lib
import go
import sgf_wrapper

import goparams

from multiprocessing.dummy import Pool as ThreadPool

# TF_RECORD_CONFIG = tf.python_io.TFRecordOptions(
#     tf.python_io.TFRecordCompressionType.ZLIB)

# The shuffle buffer size determines how far an example could end up from
# where it started; this and the interleave parameters in preprocessing can give
# us an approximation of a uniform sampling.  The default of 4M is used in
# training, but smaller numbers can be used for aggregation or validation.
# SHUFFLE_BUFFER_SIZE = int(2*1e6)
SHUFFLE_BUFFER_SIZE = goparams.SHUFFLE_BUFFER_SIZE

# Constructing tf.Examples


def _one_hot(index):
    onehot = np.zeros([go.N * go.N + 1], dtype=np.float32)
    onehot[index] = 1
    return onehot


def write_dataset(filename, dataset):
    torch.save(dataset, filename)


def read_dataset(batch_size, tf_records, num_repeats=None,
                    shuffle_records=True, shuffle_examples=True,
                    shuffle_buffer_size=None,
                    filter_amount=1.0):
    # if shuffle_buffer_size is None:
    #     shuffle_buffer_size = SHUFFLE_BUFFER_SIZE
    if shuffle_records:
        random.shuffle(tf_records)

    pool = ThreadPool()
    datasets = pool.map(torch.load, tf_records)
    pool.close()
    pool.join()
    whole_dataset = Data.ConcatDataset(datasets)


    return whole_dataset


def get_input_tensors(batch_size, tf_records, num_repeats=None,
                      shuffle_records=True, shuffle_examples=True,
                      shuffle_buffer_size=None,
                      filter_amount=0.05):
    dataset = read_dataset(batch_size, tf_records)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
    )
    return loader


def make_dataset_from_selfplay(data_extracts):
    features = []
    pis = []
    results = []
    for pos, pi, result in data_extracts:
        feature = features_lib.extract_features(pos)
        features.append(feature)
        pis.append(pi)
        results.append(result)
    torch_dataset = Data.TensorDataset(
        torch.from_numpy(np.array(features)),
        torch.from_numpy(np.array(pis)),
        torch.from_numpy(np.array(results)),
    )
    return torch_dataset


def shuffle_examples(gather_size, records_to_shuffle):
    dataset = read_dataset(gather_size, records_to_shuffle)

    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=gather_size,
        shuffle=True,
        num_workers=10,
    )
    return loader
