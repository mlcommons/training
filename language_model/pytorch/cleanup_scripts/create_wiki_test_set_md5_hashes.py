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

# Creates a text file containing md5sums from test set text file, to be used for verification of test set

import hashlib

filename = 'wiki_test_set.txt'
ofilename = 'wiki_test_set_md5.txt'

with open(filename, mode='r', newline='\n') as ifile:
  lines = ifile.read()
  articles_in_file_tmp = lines.split('\n\n')
  articles_in_file = []
  for item in articles_in_file_tmp:
    if item.rstrip() != '':
      articles_in_file.append(item)

with open(ofilename, mode='w', newline='\n') as ofile:
  for item in articles_in_file:
    ofile.write(hashlib.md5(item.encode('utf-8')).hexdigest())
    ofile.write('\n')
