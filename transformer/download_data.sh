#!/bin/bash

wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en -O tensorflow/newstest2014.en
wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de -O tensorflow/newstest2014.de

python3 data_download.py --raw_dir raw_data
