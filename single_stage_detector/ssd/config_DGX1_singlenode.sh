#!/bin/bash

## DL params
EXTRA_PARAMS=(
               --batch-size      "128"
               --warmup          "300"
             )

## System run parms
DGXNNODES=1
DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=12:00:00

## System config params
DGXNGPU=8
DGXSOCKETCORES=20
DGXNSOCKET=1
DGXHT=1 	# HT is on is 2, HT off is 1
DGXIBDEVICES=''
