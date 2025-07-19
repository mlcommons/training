# Change directory to the model directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_DIR/..

docker run -it --rm \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    --volume=/data/training:/data \
    --volume $(pwd):/workspace/code/ \
    --volume=/data/training/llama3_8b_ref/outputs:/outputs \
    --name llama-training-`whoami` rocm/mlperf:llama3_405b_training_5.1_gfx942
    # --name llama-training-`whoami` rocm/mlperf:llama31_8b_training_5.1_gfx942_v2
    # --name llama-training-`whoami` rocm/mlperf:llama31_8b_training_5.1_gfx942
    # --name llama-training-`whoami` rocm/amd-mlperf:llama2_70b_training_5.0
