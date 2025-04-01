"""Script to randomly pick certain number of text from C4 dataset.
"""

import argparse
import time
import tensorflow as tf
import tensorflow_datasets as tfds

parser = argparse.ArgumentParser(
    description="Randomly pick text from C4 dataset.")
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
    default="train",
    help="Split of dataset.")
parser.add_argument(
    "--num_examples",
    type=int,
    default=40000000,
    help="Number of examples to pick from dataset.")
parser.add_argument(
    "--output_text_file",
    type=str,
    default="",
    help="Path for output text file.")
args = parser.parse_args()

if __name__ == '__main__':
  tic = time.time()
  
  ds_name = "c4/" + args.language + ":" + args.version
  ds = tfds.load(
      ds_name,
      split=args.split,
      shuffle_files=True,
      data_dir=args.data_dir)

  num_examples = 0
  max_text_length = 0
  total_text_length = 0
  num_lines = 0
  max_line_length = 0
  total_line_length = 0
  fout = open(args.output_text_file, "wb")

  for example in ds:
    text = example["text"].numpy()
    length = len(text)
    if length > max_text_length:
      max_text_length = length
    total_text_length += length
    fout.write(text)
    fout.write(b"\n\n")

    num_examples += 1
    if (num_examples % 10000) == 0:
      print(num_examples)
   
    lines = text.split(b"\n")
    for line in lines:
      line_length = len(line)
      if line_length > max_line_length:
        max_line_length = line_length
      total_line_length += line_length
      num_lines += 1

    if num_examples >= args.num_examples:
      break

  fout.close()
  print(
      "num_examples = ", num_examples,
      "max_length = ", max_text_length,
      "avg_length = ", total_text_length / num_examples)
  print(
      "num_lines = ", num_lines,
      "max_length = ", max_line_length,
      "avg_length = ", total_line_length / num_lines)
