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

"""Utilities to create, read, write tf.Examples."""
import functools
import random

import bigtable_input
import coords
import dual_net
import features as features_lib
import go
import sgf_wrapper
import symmetries

import numpy as np
import tensorflow as tf

TF_RECORD_CONFIG = tf.python_io.TFRecordOptions(
    tf.python_io.TFRecordCompressionType.ZLIB)


def _one_hot(index):
    onehot = np.zeros([go.N * go.N + 1], dtype=np.float32)
    onehot[index] = 1
    return onehot


def make_tf_example(features, pi, value):
    """
    Args:
        features: [N, N, FEATURE_DIM] nparray of uint8
        pi: [N * N + 1] nparray of float32
        value: float
    """
    return tf.train.Example(features=tf.train.Features(feature={
        'x': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[features.tostring()])),
        'pi': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[pi.tostring()])),
        'outcome': tf.train.Feature(
            float_list=tf.train.FloatList(
                value=[value]))}))


def write_tf_examples(filename, tf_examples, serialize=True):
    """
    Args:
        filename: Where to write tf.records
        tf_examples: An iterable of tf.Example
        serialize: whether to serialize the examples.
    """
    with tf.python_io.TFRecordWriter(
            filename, options=TF_RECORD_CONFIG) as writer:
        for ex in tf_examples:
            if serialize:
                writer.write(ex.SerializeToString())
            else:
                writer.write(ex)


def batch_parse_tf_example(batch_size, layout, example_batch):
    """
    Args:
        batch_size: batch size
        layout: 'nchw' or 'nhwc'
        example_batch: a batch of tf.Example
    Returns:
        A tuple (feature_tensor, dict of output tensors)
    """
    planes = dual_net.get_features_planes()

    features = {
        'x': tf.FixedLenFeature([], tf.string),
        'pi': tf.FixedLenFeature([], tf.string),
        'outcome': tf.FixedLenFeature([], tf.float32),
    }
    parsed = tf.parse_example(example_batch, features)
    x = tf.decode_raw(parsed['x'], tf.uint8)
    x = tf.cast(x, tf.float32)

    if layout == 'nhwc':
        shape = [batch_size, go.N, go.N, planes]
    else:
        shape = [batch_size, planes, go.N, go.N]
    x = tf.reshape(x, shape)

    pi = tf.decode_raw(parsed['pi'], tf.float32)
    pi = tf.reshape(pi, [batch_size, go.N * go.N + 1])
    outcome = parsed['outcome']
    outcome.set_shape([batch_size])
    return x, {'pi_tensor': pi, 'value_tensor': outcome}


def read_tf_records(batch_size, tf_records, num_repeats=1,
                    shuffle_records=True, shuffle_examples=True,
                    shuffle_buffer_size=None, interleave=True,
                    filter_amount=1.0):
    """
    Args:
        batch_size: batch size to return
        tf_records: a list of tf_record filenames
        num_repeats: how many times the data should be read (default: One)
        shuffle_records: whether to shuffle the order of files read
        shuffle_examples: whether to shuffle the tf.Examples
        shuffle_buffer_size: how big of a buffer to fill before shuffling.
        interleave: iwhether to interleave examples from multiple tf_records
        filter_amount: what fraction of records to keep
    Returns:
        a tf dataset of batched tensors
    """
    if shuffle_examples and not shuffle_buffer_size:
        raise ValueError("Must set shuffle buffer size if shuffling examples")

    tf_records = list(tf_records)
    if shuffle_records:
        random.shuffle(tf_records)
    record_list = tf.data.Dataset.from_tensor_slices(tf_records)

    # compression_type here must agree with write_tf_examples
    map_func = functools.partial(
        tf.data.TFRecordDataset,
        buffer_size=8 * 1024 * 1024,
        compression_type='ZLIB')

    if interleave:
        # cycle_length = how many tfrecord files are read in parallel
        # The idea is to shuffle both the order of the files being read,
        # and the examples being read from the files.
        dataset = record_list.apply(tf.data.experimental.parallel_interleave(
            map_func, cycle_length=64, sloppy=True))
    else:
        dataset = record_list.flat_map(map_func)

    if filter_amount < 1.0:
        dataset = dataset.filter(
            lambda _: tf.random_uniform([]) < filter_amount)

    dataset = dataset.repeat(num_repeats)
    if shuffle_examples:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    dataset = dataset.batch(batch_size)
    return dataset


def _random_rotation(feature_layout, x_tensor, outcome_tensor):
    pi_tensor = outcome_tensor['pi_tensor']
    if feature_layout == 'nhwc':
        x_rot_tensor, pi_rot_tensor=symmetries.rotate_train_nhwc(
            x_tensor, pi_tensor)
    else:
        x_rot_tensor, pi_rot_tensor=symmetries.rotate_train_nchw(
            x_tensor, pi_tensor)

    outcome_tensor['pi_tensor'] = pi_rot_tensor
    return x_rot_tensor, outcome_tensor


def get_input_tensors(batch_size, feature_layout, tf_records, num_repeats=1,
                      shuffle_records=True, shuffle_examples=True,
                      shuffle_buffer_size=None,
                      filter_amount=0.05, random_rotation=True):
    """Read tf.Records and prepare them for ingestion by dual_net.

    See `read_tf_records` for parameter documentation.

    Returns a dict of tensors (see return value of batch_parse_tf_example)
    """
    print("Reading tf_records from {} inputs".format(len(tf_records)))
    dataset = read_tf_records(
        batch_size,
        tf_records,
        num_repeats=num_repeats,
        shuffle_records=shuffle_records,
        shuffle_examples=shuffle_examples,
        shuffle_buffer_size=shuffle_buffer_size,
        filter_amount=filter_amount,
        interleave=False)
    dataset = dataset.filter(lambda t: tf.equal(tf.shape(t)[0], batch_size))
    dataset = dataset.map(
        functools.partial(batch_parse_tf_example, batch_size, feature_layout))
    if random_rotation:
        # Unbatch the dataset so we can rotate it
        dataset = dataset.apply(tf.data.experimental.unbatch())
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            functools.partial(_random_rotation, feature_layout),
            batch_size))

    return dataset.make_one_shot_iterator().get_next()


def get_tpu_input_tensors(batch_size, feature_layout, tf_records, num_repeats=1,
                          shuffle_records=True, shuffle_examples=True,
                          shuffle_buffer_size=None,
                          filter_amount=0.05, random_rotation=True):
    # TPUs trains on sequential golden chunks to simplify preprocessing and
    # reproducibility.
    assert len(tf_records) < 101, "Use example_buffer to build a golden_chunk"

    dataset = read_tf_records(
        batch_size,
        tf_records,
        num_repeats=num_repeats,
        shuffle_records=shuffle_records,
        shuffle_examples=shuffle_examples,
        shuffle_buffer_size=shuffle_buffer_size,
        filter_amount=filter_amount,
        interleave=False)
    dataset = dataset.filter(lambda t: tf.equal(tf.shape(t)[0], batch_size))
    dataset = dataset.map(
        functools.partial(batch_parse_tf_example, batch_size, feature_layout))

    # TODO(sethtroisi@): Unify
    if random_rotation:
        # Unbatch the dataset so we can rotate it
        dataset = dataset.apply(tf.data.experimental.unbatch())
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            functools.partial(_random_rotation, feature_layout),
            batch_size, drop_remainder=True))

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def get_tpu_bt_input_tensors(games, games_nr, batch_size, feature_layout,
                             num_repeats=1,
                             number_of_games=500e3,
                             fresh_fraction=0.05,
                             random_rotation=True):
    dataset = bigtable_input.get_unparsed_moves_from_last_n_games(
        games, games_nr, number_of_games)
    dataset = dataset.repeat(num_repeats)
    dataset = dataset.batch(batch_size)
    dataset = dataset.filter(lambda t: tf.equal(tf.shape(t)[0], batch_size))
    dataset = dataset.map(
        functools.partial(batch_parse_tf_example, batch_size, feature_layout))
    if random_rotation:
        # Unbatch the dataset so we can rotate it
        dataset = dataset.apply(tf.data.experimental.unbatch())
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            functools.partial(_random_rotation, feature_layout),
            batch_size, drop_remainder=True))

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def make_dataset_from_selfplay(data_extracts):
    """
    Returns an iterable of tf.Examples.
    Args:
        data_extracts: An iterable of (position, pi, result) tuples
    """
    f = dual_net.get_features()
    tf_examples = (make_tf_example(features_lib.extract_features(pos, f),
                                   pi, result)
                   for pos, pi, result in data_extracts)
    return tf_examples


def make_dataset_from_sgf(sgf_filename, tf_record):
    pwcs = sgf_wrapper.replay_sgf_file(sgf_filename)
    tf_examples = map(_make_tf_example_from_pwc, pwcs)
    write_tf_examples(tf_record, tf_examples)


def _make_tf_example_from_pwc(position_w_context):
    f = dual_net.get_features()
    features = features_lib.extract_features(position_w_context.position, f)
    pi = _one_hot(coords.to_flat(position_w_context.next_move))
    value = position_w_context.result
    return make_tf_example(features, pi, value)
