#!/bin/bash

data_dir=${DATA_DIR:-./}
mkdir -p $data_dir

wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P $data_dir

unzip $data_dir/tiny-imagenet-200.zip -d $data_dir
