#!/bin/sh
set -ex

cd ./data/coco

if [ ! -d annotations ]; then
if [ -f coco_annotations_minival.tgz ] && [ -f annotations_trainval2014.zip ]; then
tar xzf coco_annotations_minival.tgz
unzip -n annotations_trainval2014.zip
fi
fi

if [ ! -d train2014 ]; then
if [ -f train2014.zip ]; then
unzip -n train2014.zip
fi
fi

if [ ! -d val2014 ]; then
if [ -f val2014.zip ]; then
unzip -n val2014.zip
fi
fi

