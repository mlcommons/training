#!/bin/bash
# Runs benchmark and reports time to convergence
pushd pytorch
python setup.py clean build develop --user

: "${DATA_DIR:=pytorch/datasets/coco}"
: "${OUTPUT_DIR:=pytorch/output}"

while [ $# -gt 0 ]; do
  case "$1" in
  --data_dir=*)
    DATA_DIR="${1#*=}"
    ;;
  --output_dir=*)
    OUTPUT_DIR="${1#*=}"
    ;;
  *) ;;
  esac
  shift
done

echo "DATA_DIR"
echo $DATA_DIR

DATA_ROOT_TARGET="datasets/coco"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-False}"
SOLVER_MAX_ITER="${SOLVER_MAX_ITER:-40000}"

#Link input data paths
if [ "$DATA_DIR" != "$DATA_ROOT_TARGET" ]; then
  mkdir -p $DATA_ROOT_TARGET
  ln -s $DATA_DIR/annotations $DATA_ROOT_TARGET/annotations
  ln -s $DATA_DIR/train2017 $DATA_ROOT_TARGET/train2017
  ln -s $DATA_DIR/val2017 $DATA_ROOT_TARGET/val2017
fi

pwd

# Single GPU training
time python tools/train_mlperf.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
  SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720 \
  SOLVER.STEPS "(480, 640)" SOLVER.BASE_LR 0.0025 OUTPUT_DIR $OUTPUT_DIR

popd
