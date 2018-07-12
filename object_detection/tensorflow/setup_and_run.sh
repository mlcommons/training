#!/bin/bash
set -e

pip3 install -r requirements.txt
python3 setup.py install

if [[ ! -d cocoapi ]]; then
  git clone https://github.com/cocodataset/cocoapi.git
fi

pushd cocoapi/PythonAPI
make
popd

if [[ ! -d pycocotools ]]; then
  ln -s cocoapi/PythonAPI/pycocotools/ .
fi

sudo apt-get install python3-tk


if ! (echo "$PYTHONPATH" | grep "Mask_RCNN"); then export PYTHONPATH="$PYTHONPATH:`pwd`"; fi;
python3 samples/coco/coco.py train --seed=1 --dataset="$HOME/coco_dataset" --model=imagenet --logs="$HOME/rcnn_logs" --year=2014 --download=true
