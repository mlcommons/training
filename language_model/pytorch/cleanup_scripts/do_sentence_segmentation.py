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

"""Script for sentence segmentation.

Copied and modified from https://github.com/eric-haibin-lin/text-proc.git
"""
import argparse
import glob
import io
import logging
import multiprocessing
import os
import time
import nltk

from nltk.tokenize import sent_tokenize

parser = argparse.ArgumentParser(
    description='Sentence segmentation for BERT documents.')
parser.add_argument(
    '--data',
    type=str,
    default='./*/*.compact',
    help='Input files. Default is "./*/*.compact"')
parser.add_argument(
    '--input_suffix',
    type=str,
    default='.2',
    help='Suffix for input files. Default is ".2"')
parser.add_argument(
    '--output_suffix',
    type=str,
    default='.3',
    help='Suffix for output files. Default is ".3"')
parser.add_argument(
    '--nworker',
    type=int,
    default=72,
    help='Number of workers for parallel processing.')
args = parser.parse_args()

# download package
nltk.download('punkt')

# arguments
input_files = sorted(glob.glob(os.path.expanduser(args.data)))
num_files = len(input_files)
num_workers = args.nworker
logging.basicConfig(level=logging.INFO)
logging.info('Number of input files to process = %d', num_files)


def process_one_file(one_input):
  """Separate paragraphs into sentences, for one file."""
  input_filename = one_input + args.input_suffix
  output_filename = one_input + args.output_suffix
  logging.info('Processing %s => %s', input_filename, output_filename)
  with io.open(input_filename, 'r', encoding='utf-8') as fin:
    with io.open(output_filename, 'w', encoding='utf-8') as fout:
      for line in fin:
        if len(line) == 1:
          fout.write(u'\n')
        sents = sent_tokenize(line)
        for sent in sents:
          sent_str = sent.strip()
          # if sent_str:
          fout.write('%s\n' % sent_str)
      fout.write(u'\n')


if __name__ == '__main__':
  tic = time.time()
  p = multiprocessing.Pool(num_workers)
  p.map(process_one_file, input_files)
  toc = time.time()
  logging.info('Processed %s in %.2f sec', args.data, toc - tic)
