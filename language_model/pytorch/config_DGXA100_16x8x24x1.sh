## DL params
export BATCHSIZE=24
export GRADIENT_STEPS=1
#export INIT_LOSS_SCALE=16384
export LR=0.0015
export MAX_SAMPLES_TERMINATION=7000000
export MAX_STEPS=1271
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export START_WARMUP_STEP=0
export WEIGHT_DECAY_RATE=0.01
export EVAL_ITER_START_SAMPLES=3072
export EVAL_ITER_SAMPLES=3072
export WARMUP_STEPS=100
export EXTRA_PARAMS=""
export PHASE=2

## System run parms
export DGXNNODES=16
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=04:00:00

## System config params
source ${BASH_SOURCE%/*}/config_DGXA100_common.sh
