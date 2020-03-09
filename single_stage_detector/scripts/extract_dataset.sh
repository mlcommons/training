#!/bin/sh
set -ex

cd ./data/coco

if [ ! -d train2017 ]; then
if [ -f train2017.zip ]; then
unzip -n train2017.zip
fi
fi

if [ ! -d val2017 ]; then
if [ -f val2017.zip ]; then
unzip -n val2017.zip
fi
fi

if [ ! -d annotations ]; then
if [ -f annotations_trainval2017.zip ]; then
unzip -n annotations_trainval2017.zip
fi
fi
