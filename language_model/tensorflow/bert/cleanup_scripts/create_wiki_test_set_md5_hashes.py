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
