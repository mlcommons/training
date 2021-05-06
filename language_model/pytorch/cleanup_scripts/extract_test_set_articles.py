# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import glob
import hashlib

output_filename = 'wiki_test_set.txt'
hash_file = 'wiki_test_set_md5.txt'

test_articles = []
test_hashes = []

found_hashes = []

file_glob = glob.glob('./results/part*', recursive=False)

# Load md5sums into memory
with open(hash_file, mode='r', newline='\n') as hfile:
  for line in hfile:
    test_hashes.append(line.rstrip())

assert len(test_hashes) == 500, 'Incorrect number of test set articles.'

with open(output_filename, mode='w', newline='\n') as ofile:
  for filename in file_glob:
    articles_in_file = []
    target_article_idx = None
    with open(filename, mode='r', newline='\n') as ifile:
      print("file opened:", filename)
      lines = ifile.read()
      articles_in_file_tmp = lines.split('\n\n')
      articles_in_file = []
      idx = 0
      for item in articles_in_file_tmp:
        if item.rstrip() != '':
          articles_in_file.append(item)
          if hashlib.md5(item.rstrip().encode('utf-8')).hexdigest() in test_hashes:
            print("article found at", idx)
            target_article_idx = idx
            found_hashes.append(hashlib.md5(item.rstrip().encode('utf-8')).hexdigest())
            test_articles.append(item.rstrip())
          idx += 1
    
    if not target_article_idx:
      print('article not found, continuing')
      continue

    with open(filename, mode='w', newline='\n') as ifile:
      for article in articles_in_file[:target_article_idx]:
        ifile.write(article)
        ifile.write('\n\n')

      for article in articles_in_file[target_article_idx+1:]:
        ifile.write(article)
        ifile.write('\n\n')

  if len(test_articles) != 500:
    print("Entering missing article debug section.", len(found_hashes), "hashes found.")
    missing_articles = []
    for idx, expected_hash in enumerate(test_hashes):
      if expected_hash not in found_hashes:
        missing_articles.append(idx)
        print("Missing article, reference idx:", idx)
  
  print(len(test_articles), "articles out of 500 found.")

  assert len(test_articles) == 500, 'Not all articles were found in shards. Incomplete test set.'

  for expected_hash in test_hashes:
    idx = found_hashes.index(expected_hash)
    ofile.write(test_articles[idx])
    ofile.write('\n\n')

print("n_articles =", len(test_articles))
