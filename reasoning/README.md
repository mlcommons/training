# Deepseek-v3 GRPO Reasoning Benchmark

## Overview

This benchmark trains Deepseek-v3 model for reasoning using GRPO algorithm based on Nemo-RL implementation for Nvidia GPUs.

## Container build

Docker can be build using following command in the benchmark directory

```
cd RL
docker buildx build --target release --build-context nemo-rl=. -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .
```

For more detailed information follow direct instructions from `RL/docs/docker.md` provided by Nemo-RL framewrok

## Preprocess dataset

Dataset is automatically downloaded and preprocessed on the first run. This benchmark usies OpenMathInstruct-2 hosted on Huggingface
(see https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)


## Checkpoint conversion

Benchamrk uses base pretrained checkpoint to kickstart training. Follow instruction provided by Nemo-RL about converting checkpoint into matching format. See `RL/docs/guides/deepseek.md` 

## Running experiments

A script to setup environment and run the benchmark is prepared. See `run.sub` for more information

```
export BASE_DIRECTORY=$(pwd)
export CONFIG_FILE={ONE OF POSSIBLE CONFIG PATHS HERE}
export IMAGE_FILE={HANDLE TO DOCKER IMAGE}

sbatch {SLURM specific instructions} run.sub
```

## Config selection

### DSv3
Use following configuration to train DSv3. Due to large size of the model, minimum 512 of H100 GPUs are needed to run the procedure properly

```
export CONFIG_FILE=/opt/nemo-rl/examples/configs/recipes/llm/grpo-dsv3-base-openmath.yaml
```
