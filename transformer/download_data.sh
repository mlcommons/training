#!/bin/bash

wget https://raw.githubusercontent.com/tensorflow/mlbenchmark/fixes/transformer/test_data/newstest2014.en?token=AeMkozrG27bVaqsTvxa8j3it15V2S8_2ks5a6QBUwA -O tensorflow/newstest2014.en
wget https://raw.githubusercontent.com/tensorflow/mlbenchmark/fixes/transformer/test_data/newstest2014.de?token=AeMko9qQ2IRoST3hDGUenronIm9qPTqwks5a6QDEwA -O tensorflow/newstest2014.de

python3 data_download.py --raw_dir raw_data
