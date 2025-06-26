# Lint as: python3
"""Script to clean up input wiki dump for BERT input."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import io
import logging
import multiprocessing
import os
import time


parser = argparse.ArgumentParser(description='Wiki clean up for BERT.')
parser.add_argument(
    '--data',
    type=str,
    default='./wiki_??',
    help='Input files. Default is "./wiki_??"')
parser.add_argument(
    '--input_suffix',
    type=str,
    default='',
    help='Suffix for input files. Default is ""')
parser.add_argument(
    '--output_suffix',
    type=str,
    default='.1',
    help='Suffix for output files. Default is ".1"')
parser.add_argument(
    '--nworker',
    type=int,
    default=72,
    help='Number of workers for parallel processing.')
args = parser.parse_args()


def process_one_file(one_input):
  """Remove <doc> tag and title of pages, for one file."""
  input_filename = one_input + args.input_suffix
  output_filename = one_input + args.output_suffix
  logging.info('Processing %s => %s', input_filename, output_filename)

  with io.open(input_filename, 'r', encoding='utf-8') as fin:
    with io.open(output_filename, 'w', encoding='utf-8') as fout:

      keep_next_line = True
      for line in fin:
        if not keep_next_line:
          keep_next_line = True
          continue

        if '<doc' in line:
          keep_next_line = False
          fout.write(u'\n')
          continue

        if '</doc>' in line:
          continue

        if len(line) == 1:
          continue

        # line = line.replace('<nowiki>', '').replace('</nowiki>', '')

        fout.write(line)


if __name__ == '__main__':
  input_files = sorted(glob.glob(os.path.expanduser(args.data)))
  num_files = len(input_files)
  num_workers = args.nworker
  logging.basicConfig(level=logging.INFO)
  logging.info('Number of input files to process = %d', num_files)

  tic = time.time()
  p = multiprocessing.Pool(num_workers)
  p.map(process_one_file, input_files)
  toc = time.time()
  logging.info('Processed %s in %.2f sec', args.data, toc - tic)
