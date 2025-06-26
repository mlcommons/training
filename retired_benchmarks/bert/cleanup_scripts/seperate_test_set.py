"""Script for seperating training and test sets.
"""
import argparse
import glob
import io
import logging
import multiprocessing
import os
import time
import random
import hashlib

parser = argparse.ArgumentParser(
    description='Training and test sets seperator for BERT.')
parser.add_argument(
    '--data',
    type=str,
    default='./*/*.compact',
    help='Input files. Default is "./*/*.compact"')
parser.add_argument(
    '--input_suffix',
    type=str,
    default='.3',
    help='Suffix for input files. Default is ".3"')
parser.add_argument(
    '--output_suffix',
    type=str,
    default='.4',
    help='Suffix for output training files. Default is ".4"')
parser.add_argument(
    '--nworker',
    type=int,
    default=72,
    help='Number of workers for parallel processing.')
parser.add_argument(
    '--seed',
    type=int,
    default=12345,
    help='Seed for randomization. Default is 12345.')
parser.add_argument(
    '--num_test_articles',
    type=int,
    default=10000,
    help='Number of articals withheld in test set. Default is 10k.')
parser.add_argument(
    '--test_output',
    type=str,
    default='./results/eval',
    help='Postfix for test set output. txt and md5 extensions will be added.')
args = parser.parse_args()

# arguments
input_files = sorted(glob.glob(os.path.expanduser(args.data)))
num_files = len(input_files)
num_workers = args.nworker
logging.basicConfig(level=logging.INFO)
logging.info('Number of input files to process = %d', num_files)
# test_articles_in_files = [[] for _ in range(num_files)]

def process_one_file(file_id):
  """Seperating train and eval data, for one file."""
  one_input = input_files[file_id]
  input_filename = one_input + args.input_suffix
  output_filename = one_input + args.output_suffix
  num_articles = 0
  num_tests = int((file_id+1) * args.num_test_articles * 1.0 / num_files) \
      - int(file_id * args.num_test_articles * 1.0 / num_files)
  file_seed = args.seed + file_id * 13
  rng = random.Random(file_seed)
  test_articles = []

  with io.open(input_filename, 'r', encoding='utf-8', newline='\n') as fin:
    with io.open(output_filename, 'w', encoding='utf-8', newline='\n') as fout:
      lines = fin.read()
      articles = lines.split('\n\n')
      num_articles = len(articles)
      test_article_ids = []
      while len(test_article_ids) < num_tests:
        new_id = int(rng.random() * num_articles)
        if new_id in test_article_ids:
          continue
        test_article_ids.append(new_id)

      for i in range(num_articles):
        article = articles[i]
        if i in test_article_ids:
          # test_articles_in_files[file_id].append(article)
          test_articles.append(article)
        else:
          fout.write(article)
          fout.write('\n\n')

  logging.info('Processed %s => %s, %d of %d articals picked into test set. %s',
               input_filename, output_filename, num_tests, num_articles,
               test_article_ids)
  return test_articles


if __name__ == '__main__':
  tic = time.time()
  p = multiprocessing.Pool(num_workers)
  file_ids = range(num_files)
  test_articles_in_files = p.map(process_one_file, file_ids)
  toc = time.time()
  logging.info('Processed %s (%d files) in %.2f sec',
               args.data, num_files, toc - tic)

  output_filename = args.test_output + '.txt'
  hash_filename = args.test_output + '.md5'
  with io.open(output_filename, 'w', encoding='utf-8', newline='\n') as fout:
    with io.open(hash_filename, 'w', encoding='utf-8', newline='\n') as hashout:
      for f in test_articles_in_files:
        for article in f:
          fout.write(article)
          fout.write('\n\n')
          
          article_hash = hashlib.md5(article.rstrip().encode('utf-8')).hexdigest()
          hashout.write(article_hash)
          hashout.write('\n')

