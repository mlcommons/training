#!/bin/sh
set -ex

# Get COCO 2014 data sets
mkdir -p ./data/coco
cd ./data/coco

if [ ! -d annotations ]; then
if [ ! -f coco_annotations_minival.tgz ]; then
curl -O https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz
fi
fi

if [ ! -d train2014 ]; then
if [ ! -f train2014.zip ]; then
curl -O http://images.cocodataset.org/zips/train2014.zip
fi
fi

if [ ! -d val2014 ]; then
if [ ! -f val2014.zip ]; then
curl -O http://images.cocodataset.org/zips/val2014.zip
fi
fi

if [ ! -d annotations ]; then
if [ ! -f annotations_trainval2014.zip ]; then
curl -O http://images.cocodataset.org/annotations/annotations_trainval2014.zip
fi
fi
