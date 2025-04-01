"""Script to randomly pick certain number of text from C4 dataset.
"""

import argparse
import collections
import hashlib
import io
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import time

parser = argparse.ArgumentParser(
    description="Randomly pick examples from C4 dataset.")
parser.add_argument(
    "--data_dir",
    type=str,
    default="",
    help="Path to tfds directory, which contains C4/../x.y.z.")
parser.add_argument(
    "--language",
    type=str,
    default="en",
    help="Language of dataset.")
parser.add_argument(
    "--version",
    type=str,
    default="3.0.1",
    help="Version of dataset.")
parser.add_argument(
    "--split",
    type=str,
    default="validation",
    help="Split of dataset.")
parser.add_argument(
    "--num_examples_to_pick",
    type=int,
    default=24576,
    help="Number of examples to pick from dataset.")
parser.add_argument(
    "--output_filepath",
    type=str,
    default="",
    help="Path for output tfrecord and hash files.")
args = parser.parse_args()


def create_str_feature(value):
  bytes_list = tf.train.BytesList(value=value)
  f = tf.train.Feature(bytes_list=bytes_list)
  return f


if __name__ == '__main__':
  tic = time.time()
  
  ds_name = "c4/" + args.language + ":" + args.version
  ds = tfds.load(
      ds_name,
      split=args.split,
      shuffle_files=True,
      data_dir=args.data_dir)

  num_examples = 0
  min_text_length = 2^20
  max_text_length = 0
  total_text_length = 0
  num_lines = 0
  min_line_length = 2^20
  max_line_length = 0
  total_line_length = 0
  examples = []

  for example in ds:
    examples.append(example)
    text = example["text"].numpy()
    length = len(text)
    if length < min_text_length:
      min_text_length = length
    if length > max_text_length:
      max_text_length = length
    total_text_length += length
  
    lines = text.split(b"\n")
    for line in lines:
      line_length = len(line)
      if line_length < min_line_length:
        min_line_length = line_length
      if line_length > max_line_length:
        max_line_length = line_length
      total_line_length += line_length
      num_lines += 1

    num_examples += 1
    if (num_examples % 10000) == 0:
      print(num_examples)

  print("Input:")
  print(
      "  num_examples = ", num_examples,
      "  min_length = ", min_text_length,
      "  avg_length = ", total_text_length / num_examples,
      "  max_length = ", max_text_length)
  print(
      "  num_lines = ", num_lines,
      "  min_length = ", min_line_length,
      "  avg_length = ", total_line_length / num_lines,
      "  max_length = ", max_line_length)

  min_text_length = 2^20
  max_text_length = 0
  total_text_length = 0
  num_lines = 0
  min_line_length = 2^20
  max_line_length = 0
  total_line_length = 0
 
  writer = tf.io.TFRecordWriter(args.output_filepath + ".tfrecord")
  hashout = io.open(args.output_filepath + ".txt", "w", encoding="utf-8", newline="\n")
  pick_ratio = num_examples / args.num_examples_to_pick
  num_examples_picked = 0
  i = 0
  feature_names = ["content-length", "content-type", "text", "timestamp", "url"]
  for i in range(args.num_examples_to_pick):
    example = examples[int(i * pick_ratio)]
    text = example["text"].numpy()
    length = len(text)
    if length < min_text_length:
      min_text_length = length
    if length > max_text_length:
      max_text_length = length
    total_text_length += length
    
    text_hash = hashlib.md5(text).hexdigest()
    hashout.write(text_hash)
    hashout.write('\n')

    lines = text.split(b"\n")
    for line in lines:
      line_length = len(line)
      if line_length < min_line_length:
        min_line_length = line_length
      if line_length > max_line_length:
        max_line_length = line_length
      total_line_length += line_length
      num_lines += 1
   
    features = collections.OrderedDict()
    for f in feature_names:
      features[f] = create_str_feature([example[f].numpy()])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
    num_examples_picked += 1

  writer.close()
  hashout.close()

  print("Selected:")
  print(
      "  num_examples = ", num_examples_picked,
      "  min_length = ", min_text_length,
      "  avg_length = ", total_text_length / num_examples_picked,
      "  max_length = ", max_text_length)
  print(
      "  num_lines = ", num_lines,
      "  min_length = ", min_line_length,
      "  avg_length = ", total_line_length / num_lines,
      "  max_length = ", max_line_length)


