#!/bin/bash

: "${DOWNLOAD_PATH:=/datasets/downloads/coco2017}"
: "${OUTPUT_PATH:=/datasets/coco2017}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --download-path )       shift
                                     DOWNLOAD_PATH=$1
                                     ;;
        -o | --output-path  )        shift
                                     OUTPUT_PATH=$1
                                     ;;
    esac
    shift
done

mkdir -p $DOWNLOAD_PATH
cd $DOWNLOAD_PATH
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "cced6f7f71b7629ddf16f17bbcfab6b2  ./train2017.zip"                | md5sum -c
echo "442b8da7639aecaf257c1dceb8ba8c80  ./val2017.zip"                  | md5sum -c
echo "f4bbac642086de4f52a3fdda2de5fa2c  ./annotations_trainval2017.zip" | md5sum -c

mkdir -p $OUTPUT_PATH
unzip train2017.zip -d $OUTPUT_PATH
unzip val2017.zip -d $OUTPUT_PATH
unzip annotations_trainval2017.zip -d $OUTPUT_PATH
