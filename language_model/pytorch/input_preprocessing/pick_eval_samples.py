"""Script for picking certain number of samples.
"""

import argparse
import time
import logging
import collections
import h5py
import tensorflow as tf

parser = argparse.ArgumentParser(
    description="Eval sample picker for BERT.")
parser.add_argument(
    '--input_hdf5_file',
    type=str,
    default='',
    help='Input hdf5_file path')
parser.add_argument(
    '--output_hdf5_file',
    type=str,
    default='',
    help='Output hdf5_file path')
parser.add_argument(
    '--num_examples_to_pick',
    type=int,
    default=10000,
    help='Number of examples to pick')
parser.add_argument(
    '--max_seq_length',
    type=int,
    default=512,
    help='The maximum number of tokens within a sequence.')
parser.add_argument(
    '--max_predictions_per_seq',
    type=int,
    default=76,
    help='The maximum number of predictions within a sequence.')
args = parser.parse_args()

max_seq_length = args.max_seq_length
max_predictions_per_seq = args.max_predictions_per_seq
logging.basicConfig(level=logging.INFO)

def decode_record(record):
  """Decodes a record to a TensorFlow example."""
  name_to_features = {
      "input_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "input_mask":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "segment_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "masked_lm_positions":
          tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
      "masked_lm_ids":
          tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
      "masked_lm_weights":
          tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
      "next_sentence_labels":
          tf.FixedLenFeature([1], tf.int64),
  }

  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


if __name__ == '__main__':
  tic = time.time()
  if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

  #d = tf.data.TFRecordDataset(args.input_tfrecord)
  h5_ifile = h5py.File(args.input_hdf5_file, 'r')
  
#  hdf5_compression_method = "gzip"
  hdf5_compression_method = None
  h5_writer = h5py.File(args.output_hdf5_file, 'w')

  h5_writer.create_dataset('input_ids', data=h5_ifile['input_ids'][:args.num_examples_to_pick,:], dtype='i4', compression=hdf5_compression_method)
  h5_writer.create_dataset('input_mask', data=h5_ifile['input_mask'][:args.num_examples_to_pick,:], dtype='i1', compression=hdf5_compression_method)
  h5_writer.create_dataset('segment_ids', data=h5_ifile['segment_ids'][:args.num_examples_to_pick,:], dtype='i1', compression=hdf5_compression_method)
  h5_writer.create_dataset('masked_lm_positions', data=h5_ifile['masked_lm_positions'][:args.num_examples_to_pick,:], dtype='i4', compression=hdf5_compression_method)
  h5_writer.create_dataset('masked_lm_ids', data=h5_ifile['masked_lm_ids'][:args.num_examples_to_pick,:], dtype='i4', compression=hdf5_compression_method)
  h5_writer.create_dataset('next_sentence_labels', data=h5_ifile['next_sentence_labels'][:args.num_examples_to_pick], dtype='i1', compression=hdf5_compression_method)
  h5_writer.flush()
  h5_writer.close()

  toc = time.time()
  num_examples=-1               # FIXME: This was undefined, how was it supposed to be derived?
  logging.info("Picked %d examples out of %d samples in %.2f sec",
               args.num_examples_to_pick, num_examples, toc - tic)
