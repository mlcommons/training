# 1. Problem

## Reasoning - GRPO with NeMo-Gym SWE/OpenHands.

[NeMo-RL](https://github.com/CarlosGomes98/RL) provides the implementation used for this benchmark. The benchmark trains `Qwen/Qwen3-235B-A22B-Instruct-2507` with GRPO against a NeMo-Gym software-engineering environment driven by an OpenHands SWE agent.

The relevant config files are under `RL/examples/nemo_gym`. The benchmark launch entrypoint is `RL/examples/nemo_gym/launch_nemo_gym_multinode_training.sh`, using `RL/examples/nemo_gym/grpo_qwen3_235b_swe_openhands_async.yaml`.

Formal benchmark description: <to be completed>

# 2. Directions

## Steps to configure machine

To use this repository, please ensure your system can run containers and has appropriate GPU support.

### Container setup

The Dockerfile to build for this benchmark is the NeMo-RL v0.6.0 Gym overlay Dockerfile.

```bash
cd RL
docker buildx build \
  --platform <linux/amd64 or linux/arm64> \
  -t <tag> \
  -f docker/Dockerfile.gym_v0.6.0 \
  .
```

The Dockerfile overlays the SWE/NeMo-Gym pieces on top of `nvcr.io/nvidia/nemo-rl:v0.6.0` and prefetches Gym virtual environments for `examples/nemo_gym/grpo_qwen3_235b_swe_openhands_async.yaml`.

## Steps to download and verify data

The run requires the following artifacts:

| Artifact | Description | Status |
|---|---|---|
| Policy model | Host directory containing `Qwen/Qwen3-235B-A22B-Instruct-2507`, passed as `HF_CKPT_PATH` and mounted into the container | <to be completed> |
| Megatron-Core checkpoint cache | Host directory for the HF-to-Megatron converted checkpoint cache, passed as `NRL_MEGATRON_CHECKPOINT_DIR` and mounted into the container | <to be completed> |
| Training JSONL | Host path to NeMo-Gym SWE training tasks, passed as `NEMO_GYM_SWE_TRAIN_DATA_PATH` and mounted into the container | <to be completed> |
| Validation JSONL | Host path to NeMo-Gym SWE validation tasks, passed as `NEMO_GYM_SWE_VALIDATION_DATA_PATH` and mounted into the container | <to be completed> |
| Task containers | Host directory containing Apptainer/Singularity SIF images, passed as `NEMO_GYM_SWE_SIF_DIR` and mounted into the container | <to be completed> |

To download the training and validation JSONL files using the HuggingFace CLI:

```bash
; hf download hfilaretov/Benchmark-R2E-Gym-Easy --repo-type dataset --local-dir hfilaretov__Benchmark-R2E-Gym-Easy
...

; tree hfilaretov__Benchmark-R2E-Gym-Easy
hfilaretov__Benchmark-R2E-Gym-Easy
├── benchmark_r2e_gym_easy_train.jsonl
├── benchmark_r2e_gym_easy_val.jsonl
└── README.md

1 directory, 3 files
```

The environment also requires per-task SIF images. The recipe resolves task containers from `sif_dir` with this template:

```yaml
- "${sif_dir}/r2egym_{instance_id}.sif"
```

The task containers have to be built and converted to SIF format, please see [Section 3](#data-preprocessing) below.

### Model cache setup

From outside the container, download the Hugging Face model into a host directory. The launcher requires `HF_CKPT_PATH` to point at this directory, then mounts it at `/inputs/nemo_gym/hf_ckpt` by default and passes that container path to the recipe as `policy.model_name`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install huggingface_hub

export HF_CKPT_PATH=$(pwd)/hf/Qwen/Qwen3-235B-A22B-Instruct-2507
mkdir -p "$HF_CKPT_PATH"
HF_TOKEN=<your hf token> hf download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir "$HF_CKPT_PATH"
```

The launcher also creates and mounts a host Hugging Face cache. Set `HF_HOME` before launch if you want to use a cache outside `$(pwd)/.cache`.

## Steps to run and time

All steps below are assumed to be run from this `reasoning` directory on the host; `cd RL` enters the NeMo-RL submodule checkout. The launcher submits `ray.sub` and runs training from the checkout baked into the container at `/opt/nemo-rl` by default.

```bash
cd RL

export REPO_LOCATION=$(pwd)
export EXP_NAME=<experiment name>
export CONTAINER_IMAGE_PATH=<container image path or tag>
export SLURM_ACCOUNT=<account>
export SLURM_PARTITION=<partition>
export GPUS_PER_NODE=<number of GPUs per Slurm node>
export HF_CKPT_PATH=<host path to HF checkpoint directory>
export NRL_MEGATRON_CHECKPOINT_DIR=<host path to Megatron-Core checkpoint cache directory>  # may be empty on first run
export NEMO_GYM_SWE_TRAIN_DATA_PATH=<host path to training JSONL>
export NEMO_GYM_SWE_VALIDATION_DATA_PATH=<host path to validation JSONL>
export NEMO_GYM_SWE_SIF_DIR=<host directory containing SWE task SIF images>

# Optional authentication/logging.
export HF_TOKEN=<huggingface token>
export WANDB_API_KEY=<wandb token>

# Defaults are defined by the launcher and may be overridden here.
export TRAIN_NODES=<number of training nodes>        # default: 16
export GEN_NODES=<number of generation nodes>        # default: 24
export SLURM_TIME=<walltime>                         # default: 1:0:0
export RECIPE=examples/nemo_gym/grpo_qwen3_235b_swe_openhands_async.yaml

# Optional extra mounts. The launcher automatically mounts the paths above.
export EXTRA_MOUNTS=<host_path>:<container_path>[,<host_path>:<container_path>...]

bash examples/nemo_gym/launch_nemo_gym_multinode_training.sh
```

The launcher also accepts `NODES` to override `TRAIN_NODES + GEN_NODES`, `CONTAINER_REPO_LOCATION` to override the baked checkout path `/opt/nemo-rl`, `CONTAINER_INPUT_ROOT` and the `CONTAINER_*` path variables to override the stable container mount targets, `SLURM_EXCLUDE` to exclude problematic nodes, and `SLURM_COMMENT`/`SLURM_IDLE_EXEMPT_MINS` to customize the idle-GPU reaper exemption.

# 3. Dataset/Environment

### Publication/Attribution

We use a subset of the [R2E-Gym/R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) dataset.

### Data preprocessing

The recipe consumes prebuilt JSONL files through from [Benchmark-R2E-Gym-Easy](https://huggingface.co/datasets/hfilaretov/Benchmark-R2E-Gym-Easy).
Each row represents a software-engineering task for the NeMo-Gym environment.

The JSONL files refer to SIF container files that need to be generated.
This is a two-step process:
1. Images are built from the repository and git revision specified in the dataset.
2. These images are converted to SIF file format.

You can build the container defined in `./dataset-processing-container` that already pre-packages all necessary dependencies and can be used for both steps.

You should be able to run the following:

Prepare the builder image:

```bash
cd RL/docker/dataset-processing-container
export DOCKER_REGISTRY=<your-container-registry>
docker build --push -t $DOCKER_REGISTRY/grpo-data-builder:latest .
```

Note: to build the dataset images within the builder image, you need to mount the Docker daemon socket inside the container.
If you do not want to do that, please set up an environment equivalent to the builder image, and then run the scripts outside a container.

To build the images and push them to a registry, run on a host that has your target architecture to build natively:

```bash
export HF_TOKEN=<read-token-for-huggingface>
export DOCKER_REGISTRY=<url-to-docker-registry>
export DOCKER_TOKEN=<docker-registry-token>
export DOCKER_USER=<docker-registry-username>
export STATE_DIR=<path-to-persistent-storage>
export MAX_WORKERS=<maximum-number-of-parallel-build-tasks>

# R2E-Gym Easy subset
docker run -it --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $STATE_DIR:/workspace/state \
    -e DOCKER_REGISTRY -e DOCKER_TOKEN -e DOCKER_USER -e HF_TOKEN -e MAX_WORKERS \
    $DOCKER_REGISTRY/grpo-data-builder:latest \
    /workspace/run-r2e-gym-build-images.sh
```

To convert the images from the registry to SIF files:

```bash
export SIF_LOCAL_DIR=<local-directory-to-store-sif-containers>

# SIF images, final dataset
docker run -it --rm \
    -v $SIF_LOCAL_DIR:/opt/data \
    -e DOCKER_REGISTRY -e DOCKER_TOKEN -e DOCKER_USER -e HF_TOKEN -e MAX_WORKERS \
    $DOCKER_REGISTRY/grpo-data-builder:latest \
    /workspace/run-build-sif-images.sh
```

Please note that the above container will use its local storage to build the SIF files and then copy them over to your `SIF_LOCAL_DIR`.
You therefore might be constrained in the number of `$MAX_WORKERS` by your available local storage.

### Training and test data separation

The config uses separate training and validation JSONL files:

```yaml
policy:
  model_name: ${oc.env:HF_CKPT_PATH}
data:
  train:
    data_path: ${oc.env:NEMO_GYM_SWE_TRAIN_DATA_PATH}
  validation:
    data_path: ${oc.env:NEMO_GYM_SWE_VALIDATION_DATA_PATH}
sif_dir: ${oc.env:NEMO_GYM_SWE_SIF_DIR}
```

The official split procedure is <to be completed>.

### Training data order

Training data order is preserved by the recipe with `data.shuffle: false`. The intended ordering or curriculum of the training JSONL is <to be completed>.

### Test data order

Validation data order is preserved by the recipe. The config uses `grpo.max_val_samples: null`, so validation thoroughness is inferred from the validation dataset size unless overridden.

### Simulation environment (RL models only)

The benchmark uses NeMo-Gym with the SWE/OpenHands agent configuration. Rollouts are collected through a vLLM-backed policy server, with OpenHands interacting with task containers via Apptainer/Singularity. The async recipe uses non-colocated generation and training, with one-step-stale trajectories corrected by importance sampling.

# 4. Model

### Publication/Attribution

Model and implementation attribution: <to be completed>

### List of layers

| **Component** | **Architecture** | **Parameters** | **Technical Details** |
|---------------|------------------|----------------|-----------------------|
| **Policy model** | Qwen3 MoE Transformer | 235B total, 22B active | `Qwen/Qwen3-235B-A22B-Instruct-2507` |
| **Training runtime** | Megatron-Core through NeMo-RL | Same policy weights | TP4 x CP4 x PP8 minimum training replica; expert model parallel size 16 |
| **Generation runtime** | vLLM | Same policy weights | TP16, BF16, HTTP server exposed for NeMo-Gym |
| **SWE environment** | NeMo-Gym + OpenHands | N/A | Agent max turns 15 |

Exact layer-by-layer model description: <to be completed>

### Weight and bias initialization

Training starts from a pretrained Hugging Face checkpoint converted to Megatron-Core format. Random initialization is not used for the policy model. Any benchmark-specific initialization details are <to be completed>.

### Loss function

The recipe uses token-level GRPO with reward normalization and a leave-one-out baseline. Reference-policy KL is disabled (`reference_policy_kl_penalty: 0`), and the async recipe uses importance-sampling correction for one-step-stale rollouts.

### Optimizer

Adam with distributed optimizer state. The async recipe sets `lr: 5.0e-6`, `weight_decay: 0.0`, BF16 training, and FP32 optimizer parameters.

### Precision

The recipe uses BF16 policy precision by default.

# 5. Quality

### Quality metric

The checkpointing metric is `val:total_reward/mean`, computed from NeMo-Gym validation rollouts.

### Quality target

<to be completed>

### Evaluation frequency

The async recipe sets `grpo.val_period: 5`, `grpo.val_at_start: true`, and `grpo.val_at_end: true`.

### Evaluation thoroughness

The validation set size is <to be completed>.
