#!/bin/bash

python3 data_download.py --raw_dir raw_data

wget https://raw.githubusercontent.com/tensorflow/models/master/official/transformer/test_data/newstest2014.en -O ./raw_data/newstest2014.en
wget https://raw.githubusercontent.com/tensorflow/models/master/official/transformer/test_data/newstest2014.de -O ./raw_data/newstest2014.de
