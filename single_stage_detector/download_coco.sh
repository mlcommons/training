#!/bin/bash

# Get COCO 2017 data sets
COCO_DIR=${1:-"/datasets/coco2017"}
dir=$(pwd)
mkdir $COCO_DIR; cd $COCO_DIR
curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
cd $dir
