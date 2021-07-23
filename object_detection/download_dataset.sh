#!/bin/bash

# Get COCO 2017 data sets
DATA_ROOT_DIR="${DATA_ROOT_DIR:-./pytorch/datasets/coco}"
echo "Downloaading to folder: $DATA_ROOT_DIR"
mkdir -p $DATA_ROOT_DIR
pushd $DATA_ROOT_DIR

curl -O https://dl.fbaipublicfiles.com/detectron/coco/coco_annotations_minival.tgz
echo "Extracting coco_annotations_minival.tgz ..."
tar -xzf coco_annotations_minival.tgz &>/dev/null

curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "Extracting annotations_trainval2017.zip ..."
n_files=`unzip -l  annotations_trainval2017.zip| grep .json | wc -l`
unzip annotations_trainval2017.zip | pv -l -s $n_files > /dev/null

curl -O http://images.cocodataset.org/zips/val2017.zip
echo "Extracting val2017.zip ..."
n_files=`unzip -l  val2017.zip| grep .jpg | wc -l`
unzip val2017.zip | pv -l -s $n_files > /dev/null

curl -O http://images.cocodataset.org/zips/train2017.zip
echo "Extracting train2017.zip ..."
n_files=`unzip -l  train2017.zip| grep .jpg | wc -l`
unzip train2017.zip | pv -l -s $n_files > /dev/null

# TBD: MD5 verification
# $md5sum *.zip *.tgz
#f4bbac642086de4f52a3fdda2de5fa2c  annotations_trainval2017.zip
#cced6f7f71b7629ddf16f17bbcfab6b2  train2017.zip
#442b8da7639aecaf257c1dceb8ba8c80  val2017.zip
#2d2b9d2283adb5e3b8d25eec88e65064  coco_annotations_minival.tgz

popd
