#!/bin/bash


DATA_DIR="${DATA_DIR:-pytorch/datasets/coco}"
DATA_ROOT_TARGET="pytorch/datasets/coco"
OUTPUT_DIR="${OUTPUT_DIR:-}"

SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-}"
SOLVER_MAX_ITER="${SOLVER_MAX_ITER:-40000}"

#Link input data paths
if [ "$DATA_DIR" != "$DATA_ROOT_TARGET" ]; then
       mkdir -p $DATA_ROOT_TARGET
       ln -s $DATA_DIR/annotations $DATA_ROOT_TARGET/annotations
       ln -s $DATA_DIR/train2017 $DATA_ROOT_TARGET/train2017
       ln -s $DATA_DIR/test2017 $DATA_ROOT_TARGET/test2017
       ln -s $DATA_DIR/val2017 $DATA_ROOT_TARGET/val2017
       echo $DATA_ROOT_TARGET
       ls -lah $DATA_ROOT_TARGET
fi

# Runs benchmark and reports time to convergence
pushd pytorch
# Single GPU training
time python tools/train_mlperf.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
       SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" \
       SOLVER.MAX_ITER $SOLVER_MAX_ITER SOLVER.BASE_LR 0.0025 SAVE_CHECKPOINTS $SAVE_CHECKPOINTS OUTPUT_DIR $OUTPUT_DIR
       
popd
