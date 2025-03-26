## DL params
export BATCHSIZE=48
export GRADIENT_STEPS=1
#export INIT_LOSS_SCALE=16384
export LR=0.00035221
export MAX_SAMPLES_TERMINATION=10000000
export MAX_STEPS=8000
export OPT_LAMB_BETA_1=0.91063
export OPT_LAMB_BETA_2=0.96497
export START_WARMUP_STEP=0
export WEIGHT_DECAY_RATE=0.01
export EVAL_ITER_START_SAMPLES=768
export EVAL_ITER_SAMPLES=768
export WARMUP_STEPS=420
export EXTRA_PARAMS=""
export PHASE=2

## System run parms
export DGXNNODES=2
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=04:00:00

## System config params
source ${BASH_SOURCE%/*}/config_DGXA100_common.sh
