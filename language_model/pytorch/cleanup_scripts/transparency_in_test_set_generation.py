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

output_filename = 'wiki_test_set.txt'

test_articles = []

file_glob = glob.glob('./results/part*', recursive=False)

with open(output_filename, mode='w', newline='\n') as ofile:
  for filename in file_glob:
    articles_in_file = []
    with open(filename, mode='r', newline='\n') as ifile:
      lines = ifile.read()
      articles_in_file_tmp = lines.split('\n\n')
      articles_in_file = []
      for item in articles_in_file_tmp:
        if item.rstrip() != '':
          articles_in_file.append(item)
    
    target_article = min(42, len(articles_in_file) // 2)
    test_articles.append(articles_in_file[target_article])

    with open(filename, mode='w', newline='\n') as ifile:
      for article in articles_in_file[:target_article]:
        ifile.write(article)
        ifile.write('\n\n')

      for article in articles_in_file[target_article+1:]:
        ifile.write(article)
        ifile.write('\n\n')

  for article in test_articles:
    ofile.write(article)
    ofile.write('\n\n')

print("n_articles =", len(test_articles))
