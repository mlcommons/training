#!/bin/sh
set -ex

# Get COCO 2017 data sets
mkdir -p ./data/coco
cd ./data/coco

if [ ! -d train2017 ]; then
if [ ! -f train2017.zip ]; then
curl -O http://images.cocodataset.org/zips/train2017.zip
fi
fi

if [ ! -d val2017 ]; then
if [ ! -f val2017.zip ]; then
curl -O http://images.cocodataset.org/zips/val2017.zip
fi
fi

if [ ! -d annotations ]; then
if [ ! -f annotations_trainval2017.zip ]; then
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
fi
fi
