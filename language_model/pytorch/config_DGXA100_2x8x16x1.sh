## DL params
export BATCHSIZE=16
export GRADIENT_STEPS=1
export LR=3.5e-4
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=13700
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0
export WARMUP_STEPS=0.0
export WEIGHT_DECAY_RATE=0.01
export EVAL_ITER_START_SAMPLES=256
export EVAL_ITER_SAMPLES=256
export EXTRA_PARAMS=""
export PHASE=2
## System run parms
export DGXNNODES=2
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=04:00:00

## System config params
source ${BASH_SOURCE%/*}/config_DGXA100_common.sh
