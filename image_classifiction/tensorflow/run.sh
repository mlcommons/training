# TODO

set -e

export PYTHONPATH=`pwd`:$PYTHONPATH
#python3 official/resnet/imagenet_main.py --data_dir ./imn/imagenet/combined/ --model_dir /tmp/imn_example --train_epochs 10000 --stop_threshold 0.749 --batch_size 64 --version 1 --resnet_size 50
python3 official/resnet/imagenet_main.py --data_dir /imn/imagenet/combined/ --model_dir /tmp/imn_example --train_epochs 10000 --stop_threshold 0.2 --batch_size 64 --version 1 --resnet_size 50
