## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths need to be specified based on the current system config
export DATADIR="/lustre/fsw/mlperf/mlperft-bert/hdf5/v1p0_ref/2048_shards_uncompressed"
export EVALDIR="/lustre/fsw/mlperf/mlperft-bert/hdf5/v1p0_ref/eval_set_uncompressed"
export DATADIR_PHASE2="/lustre/fsw/mlperf/mlperft-bert/hdf5/v1p0_ref/2048_shards_uncompressed"
export CHECKPOINTDIR="$CI_BUILDS_DIR/$SLURM_ACCOUNT/$CI_JOB_ID/ci_checkpoints"
export CHECKPOINTDIR_PHASE1="/raid/datasets/bert/checkpoints/checkpoint_phase1"

# The directory isn't required
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"

