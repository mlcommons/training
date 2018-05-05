#!/bin/bash

wget https://raw.githubusercontent.com/tensorflow/mlbenchmark/master/transformer/test_data/newstest2014.de?token=ABCvMNi6FA6_WuLcXj_UJLv_GUc8I9NWks5a9yfEwA -O tensorflow/newstest2014.en
wget https://raw.githubusercontent.com/tensorflow/mlbenchmark/master/transformer/test_data/newstest2014.en?token=ABCvMOlxzi69iyZbqRXa0tVuRzSz83v9ks5a9yfIwA -O tensorflow/newstest2014.de

python3 data_download.py --raw_dir raw_data
