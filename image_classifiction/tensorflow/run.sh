#/bin/bash

RANDOM_SEED=$1
QUALITY=$2
set -e

export PYTHONPATH=`pwd`:$PYTHONPATH

python3 official/resnet/imagenet_main.py $RANDOM_SEED --data_dir /imn/imagenet/combined/ --model_dir /tmp/imn_example --train_epochs 10000 --stop_threshold $QUALITY --batch_size 64 --version 1 --resnet_size 50
