# Container build

Docker can be build using

```
cd RL
docker buildx build --target release --build-context nemo-rl=. -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .
```

For more information follow instructions from `RL/docs/docker.md`


# Running experiments

A script to setup environment and run the benchmark is prepared. See `run.sub` for more information

```
export BASE_DIRECTORY=$(pwd)
export CONFIG_FILE={ONE OF POSSIBLE CONFIG PATHS HERE}
export IMAGE_FILE={HANDLE TO DOCKER IMAGE}

sbatch {SLURM specific instructions} run.sub
```

## Config selection

### Qwen3-30B-A3B

```
export CONFIG_FILE=/opt/nemo-rl/examples/configs/recipes/llm/grpo-qwen3-30ba3b-base-openmath.yaml
```


### DSv3
Additional steps to convert checkpoints are needed. See `RL/docs/guides/deepseek.md`

```
export CONFIG_FILE=/opt/nemo-rl/examples/configs/recipes/llm/grpo-dsv3-base-openmath.yaml
```
