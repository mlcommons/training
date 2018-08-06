# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
"""Translate text or files using trained transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin

from utils.dataset import VOCAB_FILE
from model import model_params
from utils import tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F

_DECODE_BATCH_SIZE = 32
_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6



def _get_sorted_inputs(filename):
  """Read and sort lines from the file sorted by decreasing length.

  Args:
    filename: String name of file to read inputs from.
  Returns:
    Sorted list of inputs, and dictionary mapping original index->sorted index
    of each element.
  """
  with open(filename) as f:
    records = f.read().split("\n")
    inputs = [record.strip() for record in records]
    if not inputs[-1]:
      inputs.pop()

  input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)

  sorted_inputs = []
  sorted_keys = {}
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys


def _encode_and_add_eos(line, subtokenizer):
  """Encode line with subtokenizer, and add EOS id to the end."""
  return subtokenizer.encode(line) + [tokenizer.EOS_ID]


def _trim_and_decode(ids, subtokenizer):
  """Trim EOS and PAD tokens from ids, and decode to return a string."""
  try:
    index = list(ids).index(tokenizer.EOS_ID)
    return subtokenizer.decode(ids[:index])
  except ValueError:  # No EOS found in sequence
    return subtokenizer.decode(ids)


def translate_file(
    model, subtokenizer, input_file, output_file=None,
    print_all_translations=True, device=None):
  """Translate lines in file, and save to output file if specified.

  Args:
    estimator: tf.Estimator used to generate the translations.
    subtokenizer: Subtokenizer object for encoding and decoding source and
       translated lines.
    input_file: file containing lines to translate
    output_file: file that stores the generated translations.
    print_all_translations: If true, all translations are printed to stdout.

  Raises:
    ValueError: if output file is invalid.
  """
  batch_size = _DECODE_BATCH_SIZE

  # Read and sort inputs by length. Keep dictionary (original index-->new index
  # in sorted list) to write translations in the original order.
  sorted_inputs, sorted_keys = _get_sorted_inputs(input_file)
  num_decode_batches = (len(sorted_inputs) - 1) // batch_size + 1

  def pad_batch(batch):
    max_len = 0
    for example in batch:
      max_len = max(max_len, example.shape[0])
    padded_batch = [F.pad(example, [0, max_len - example.shape[0]])
        for example in batch]
    padded_batch = torch.stack(padded_batch)
    return padded_batch

  def input_generator():
    """Yield encoded strings from sorted_inputs."""
    for i, line in enumerate(sorted_inputs):
      if i % batch_size == 0:
        batch_num = (i // batch_size) + 1

        print("Decoding batch %d out of %d." % (batch_num, num_decode_batches))
      encoded_line = _encode_and_add_eos(line, subtokenizer)
      encoded_line = torch.tensor(encoded_line).int()
      encoded_line = encoded_line.to(model.device)
      yield encoded_line

  def input_fn():
    """Created batched dataset of encoded inputs."""
    batch = []
    for sample in input_generator():
      batch.append(sample)
      if len(batch) == batch_size:
        yield pad_batch(batch)
        batch = []
    if len(batch) > 0:
      yield pad_batch(batch)

  translations = []

  for i, batch in enumerate(input_fn()):
    prediction = model.predict(batch)
    for splitted_prediction in prediction["outputs"]:
      translation = _trim_and_decode(splitted_prediction, subtokenizer)
      translations.append(translation)

      if print_all_translations:
        print("Translating:")
        print("\tInput: %s" % sorted_inputs[i])
        print("\tOutput: %s\n" % translation)
        print("=" * 100)

  # Write translations in the order they appeared in the original file.
  if output_file is not None:
    if os.path.isdir(output_file):
      raise ValueError("File output is a directory, will not save outputs to "
                       "file.")
    print("Writing to file %s" % output_file)
    with open(output_file, "w") as f:
      for index in range(len(sorted_keys)):
        f.write("%s\n" % translations[sorted_keys[index]])

