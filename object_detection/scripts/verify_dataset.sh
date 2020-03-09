#!/bin/sh
set -ex

cd ./data/coco

if [ -f coco_annotations_minival.tgz ]; then
ACTUAL_TEST=`cat coco_annotations_minival.tgz | md5sum`
EXPECTED_TEST='2d2b9d2283adb5e3b8d25eec88e65064  -'
if [[ "$ACTUAL_TEST" = "$EXPECTED_TEST" ]]; then
  echo "OK: correct coco_annotations_minival.tgz"
else
  echo "ERROR: incorrect coco_annotations_minival.tgz"
  echo "ERROR: expected $EXPECTED_TEST"
  echo "ERROR: found $ACTUAL_TEST"
fi
fi

if [ -f annotations_trainval2014.zip ]; then
ACTUAL_TEST=`cat annotations_trainval2014.zip | md5sum`
EXPECTED_TEST='0a379cfc70b0e71301e0f377548639bd  -'
if [[ "$ACTUAL_TEST" = "$EXPECTED_TEST" ]]; then
  echo "OK: correct annotations_trainval2014.zip"
else
  echo "ERROR: incorrect annotations_trainval2014.zip"
  echo "ERROR: expected $EXPECTED_TEST"
  echo "ERROR: found $ACTUAL_TEST"
fi
fi

if [ -f train2014.zip ]; then
ACTUAL_TEST=`cat train2014.zip | md5sum`
EXPECTED_TEST='0da8c0bd3d6becc4dcb32757491aca88  -'
if [[ "$ACTUAL_TEST" = "$EXPECTED_TEST" ]]; then
  echo "OK: correct train2014.zip"
else
  echo "ERROR: incorrect train2014.zip"
  echo "ERROR: expected $EXPECTED_TEST"
  echo "ERROR: found $ACTUAL_TEST"
fi
fi

if [ -f val2014.zip ]; then
ACTUAL_TEST=`cat val2014.zip | md5sum`
EXPECTED_TEST='a3d79f5ed8d289b7a7554ce06a5782b3  -'
if [[ "$ACTUAL_TEST" = "$EXPECTED_TEST" ]]; then
  echo "OK: correct val2014.zip"
else
  echo "ERROR: incorrect val2014.zip"
  echo "ERROR: expected $EXPECTED_TEST"
  echo "ERROR: found $ACTUAL_TEST"
fi
fi
