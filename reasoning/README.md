# 1. Problem

## SWE Agent Reinforcement Learning  - GRPO with NeMo-Gym SWE/OpenHands.

[NeMo-RL](https://github.com/CarlosGomes98/RL/tree/mlperf-training-qwen35) provides the implementation used for this benchmark from branch `mlperf-training-qwen35` at commit `e4d0b38c3e9146ebf055647c85d305994d2bdb42`. The benchmark uses reinforcement learning to train `Qwen/Qwen3.5-397B-A17B` with GRPO against a NeMo-Gym software-engineering environment driven by an OpenHands SWE agent.

The task is to improve the SWE agent's accuracy in solving held-out R2E-Gym software-engineering tasks. A rollout receives reward 1 when the generated patch passes the task evaluation and reward 0 otherwise.

The relevant config files are under `RL/examples/nemo_gym` and `RL/qwen_35`. The benchmark launch entrypoint is `RL/examples/nemo_gym/launch_qwen35_nemo_gym_multinode_training.sh`, using `RL/qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_benchmark.yaml`.

# 2. Directions

## Steps to configure machine

To use this repository, please ensure your have access to a SLURM cluster with Enroot/Pyxis and at least 64x4 GB200 GPUs.

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

The Dockerfile overlays the SWE/NeMo-Gym pieces on top of `nvcr.io/nvidia/nemo-rl:v0.6.0` and prefetches Gym virtual environments for `qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml`.

## Steps to download and verify data

The run requires the following artifacts:

| Artifact | Description | Status |
|---|---|---|
| Policy model | Host directory containing `Qwen/Qwen3.5-397B-A17B`, passed as `HF_CKPT_PATH`, mounted into the container, and exposed to the recipe through `CONTAINER_HF_CKPT_PATH` | Download from Hugging Face |
| Megatron-Core checkpoint cache | Host directory for the HF-to-Megatron converted checkpoint cache, passed as `NRL_MEGATRON_CHECKPOINT_DIR` and mounted into the container | Empty directory is allowed on first run |
| Training JSONL | Host path to NeMo-Gym SWE training tasks, passed as `NEMO_GYM_SWE_TRAIN_DATA_PATH` and mounted into the container | Download `benchmark_r2e_gym_easy_train.jsonl` or rebuild with `RL/tools/create_r2e_gym_easy_subset_jsonl.py` |
| Validation JSONL | Host path to NeMo-Gym SWE validation tasks, passed as `NEMO_GYM_SWE_VALIDATION_DATA_PATH` and mounted into the container | Download `benchmark_r2e_gym_easy_val.jsonl` or rebuild with `RL/tools/create_r2e_gym_easy_subset_jsonl.py` |
| Task containers | Host directory containing Apptainer/Singularity SIF images in the layout expected by the recipe, passed as `NEMO_GYM_SWE_SIF_DIR` and mounted into the container | Build with `RL/docker/dataset-processing-container` |

To download the training and validation JSONL files using the HuggingFace CLI:

```bash
hf download hfilaretov/Benchmark-R2E-Gym-Easy --repo-type dataset --local-dir hfilaretov__Benchmark-R2E-Gym-Easy
...

tree hfilaretov__Benchmark-R2E-Gym-Easy
hfilaretov__Benchmark-R2E-Gym-Easy
├── benchmark_r2e_gym_easy_train.jsonl
├── benchmark_r2e_gym_easy_val.jsonl
└── README.md

1 directory, 3 files
```

The environment also requires per-task SIF images. The recipe resolves task containers from `sif_dir` with this template:

```yaml
- "${sif_dir}/r2egym/{instance_id}.sif"
```

The task containers have to be built and converted to SIF format, please see [Section 3](#data-preprocessing) below.

### Model cache setup

From outside the container, download the Hugging Face model into a host directory. The launcher requires `HF_CKPT_PATH` to point at this directory, mounts that path into the container, and exports `CONTAINER_HF_CKPT_PATH` to the recipe as `policy.model_name`. By default, `CONTAINER_HF_CKPT_PATH` is the same path as `HF_CKPT_PATH`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install huggingface_hub

export HF_CKPT_PATH=$(pwd)/hf/Qwen/Qwen3.5-397B-A17B
mkdir -p "$HF_CKPT_PATH"
HF_TOKEN=<your hf token> hf download Qwen/Qwen3.5-397B-A17B --local-dir "$HF_CKPT_PATH"
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
export MLPERF_TARGET_ACCURACY=<target reward mean>  # default: 1.0 until the target is finalized
export GRPO_SEED=<integer seed>                      # default: random per launch

# Defaults are defined by the launcher and may be overridden here.
export TRAIN_NODES=<number of training nodes>        # default: 16
export GEN_NODES=<number of generation nodes>        # default: 24
export SLURM_TIME=<walltime>                         # default: 1:0:0
export RECIPE=qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_benchmark.yaml

# Optional extra mounts. The launcher automatically mounts the paths above.
export EXTRA_MOUNTS=<host_path>:<container_path>[,<host_path>:<container_path>...]

bash examples/nemo_gym/launch_qwen35_nemo_gym_multinode_training.sh
```

The launcher also accepts `NODES` to override `TRAIN_NODES + GEN_NODES`, `CONTAINER_REPO_LOCATION` to override the baked checkout path `/opt/nemo-rl`, `CONTAINER_INPUT_ROOT` and the `CONTAINER_*` path variables to override container-side paths.

# 3. Dataset/Environment

### Publication/Attribution

We use a subset of the [R2E-Gym/R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset) dataset.

### Data preprocessing

The recipe consumes prebuilt JSONL files from [Benchmark-R2E-Gym-Easy](https://huggingface.co/datasets/hfilaretov/Benchmark-R2E-Gym-Easy).
Each row represents a software-engineering task for the NeMo-Gym environment. We filtered the original `R2E-Gym/R2E-Gym-Subset` dataset based on these two conditions:
* whether an environment container image successfully builds for both x86_64 and aarch64
* complexity using the following condition:
  ```
  where num_non_test_func_methods == 1 | where num_non_test_files == 1 | where num_non_test_lines <= 20
  ```

To build the JSONL files yourself, run the converter from the RL checkout:

```bash
cd RL

# Optional token
export HF_TOKEN=<read-token>
hf download R2E-Gym/R2E-Gym-Subset --repo-type dataset --local-dir tmp/R2E-Gym__R2E-Gym-Subset
uv run --with pyarrow python tools/create_r2e_gym_easy_subset_jsonl.py \
  --dataset-dir tmp/R2E-Gym__R2E-Gym-Subset \
  --output-dir outputs/data/ \
  --cache-dir tmp/r2e_repo_cache \
  --train-ids tools/train-instance-ids.txt \
  --val-ids tools/val-instance-ids.txt
```

You'll have the relevant output files in `outputs`:

```bash
wc -l outputs/data/benchmark_r2e_gym_easy_train.jsonl \
      outputs/data/benchmark_r2e_gym_easy_val.jsonl \
      outputs/data/r2e_gym_subset_full.jsonl
     721 outputs/data/benchmark_r2e_gym_easy_train.jsonl
     256 outputs/data/benchmark_r2e_gym_easy_val.jsonl
    4578 outputs/data/r2e_gym_subset_full.jsonl
    5555 total
```

The JSONL files refer to SIF container files that need to be generated.
This is a two-step process:
1. Images are built from the repository and git revision specified in the dataset.
2. These images are converted to SIF file format.

You can build the container defined in `RL/docker/dataset-processing-container` that already pre-packages all necessary dependencies and can be used for both steps.

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
  model_name: ${oc.env:CONTAINER_HF_CKPT_PATH}
data:
  train:
    data_path: ${oc.env:NEMO_GYM_SWE_TRAIN_DATA_PATH}
  validation:
    data_path: ${oc.env:NEMO_GYM_SWE_VALIDATION_DATA_PATH}
sif_dir: ${oc.env:NEMO_GYM_SWE_SIF_DIR}
```

The official split is defined by the fixed instance-id files `RL/tools/train-instance-ids.txt` and `RL/tools/val-instance-ids.txt`. The conversion script validates that the lists do not overlap, writes matching rows to `benchmark_r2e_gym_easy_train.jsonl` and `benchmark_r2e_gym_easy_val.jsonl`, and leaves rows in neither list only in `r2e_gym_subset_full.jsonl`.

### Training data order

Training data order is preserved by the recipe with `data.shuffle: false`. The converter writes the training JSONL in the order encountered in the converted R2E-Gym subset after filtering by `RL/tools/train-instance-ids.txt`; the benchmark does not add runtime shuffling.

### Test data order

Validation data order is preserved by the recipe. The config uses `grpo.max_val_samples: null`, so validation thoroughness is inferred from the validation dataset size unless overridden. The benchmark does not add runtime shuffling of validation data.

### Simulation environment (RL models only)

The benchmark uses NeMo-Gym with the SWE/OpenHands agent configuration. Rollouts are collected through a vLLM-backed policy server, with OpenHands interacting with task containers via Apptainer/Singularity. The async recipe uses non-colocated generation and training, with one-step-stale trajectories corrected by importance sampling.

# 4. Model

### Publication/Attribution

The policy starts from the [`Qwen/Qwen3.5-397B-A17B`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) checkpoint released by the Qwen team. The reference training implementation is NeMo-RL with Qwen 3.5 support from the `mlperf-training-qwen35` branch.

### Model details

Architecture values below are taken from the [Hugging Face model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) and [`config.json`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json).

| Config | Value |
| :-- | :-- |
| # Total Parameters | 397B |
| # Active Parameters | 17B |
| # Layers | 60 |
| Hidden Layout | 3 Gated DeltaNet + 1 Gated Attention layers per block (15 blocks) |
| Attention Type | Hybrid Gated DeltaNet + Gated Attention |
| Gated DeltaNet Heads (V / QK) | 64 / 16 |
| Gated DeltaNet Head Dimension | 128 |
| Gated Attention Heads (Q / KV) | 32 / 2 |
| Gated Attention Head Dimension | 256 |
| RoPE Dimension | 64 |
| Model Dimension | 4,096 |
| # Routed Experts | 512 |
| # Active Routed Experts | 10 |
| # Shared Experts | 1 |
| Expert Intermediate Dimension | 1,024 |
| Activation | SiLU (SwiGLU in MoE) |
| Normalization | RMSNorm |
| Vocab Size | 248,320 |
| Native Context Length | 262,144 |
| Benchmark Context Length | 65,536 |

### Benchmark runtime

| **Component** | **Architecture** | **Parameters** | **Technical Details** |
|---------------|------------------|----------------|-----------------------|
| **Training runtime** | Megatron-Core through NeMo-RL | Same policy weights | TP4 x PP2 x CP1, EP16, BF16 |
| **Generation runtime** | vLLM | Same policy weights | TP8, EP8, 64k benchmark context, HTTP server exposed for NeMo-Gym |
| **SWE environment** | NeMo-Gym + OpenHands | N/A | CodeActAgent, max 30 turns |

### Weight and bias initialization

Training starts from the pretrained Hugging Face checkpoint converted to Megatron-Core format. Random initialization is not used for the policy model. The first run can populate `NRL_MEGATRON_CHECKPOINT_DIR` with the converted checkpoint cache.

MoE router weights are kept frozen.

### Loss function

The recipe uses token-level GRPO with reward normalization and a leave-one-out baseline. Reference-policy KL is disabled (`reference_policy_kl_penalty: 0`), and the async recipe uses importance-sampling correction for one-step-stale rollouts.

### Optimizer

AdamW with distributed optimizer state.

| Parameter | Value |
| :-- | :-- |
| Optimizer | AdamW |
| Base learning rate | `2.0e-6` |
| End learning rate | `2.0e-6` |
| Learning-rate schedule | Constant |
| Warmup steps | 2 |
| Weight decay | `0.0` |
| Adam beta1 | `0.9` |
| Adam beta2 | `0.999` |
| Adam epsilon | `1e-8` |
| Gradient clipping | `1.0` |
| Distributed optimizer | Enabled |
| Optimizer parameters | FP32 |
| Training precision | BF16 |

### Precision

The recipe uses BF16 policy precision by default.

# 5. Quality

### Quality metric

The quality metric is `val:accuracy`, computed from NeMo-Gym validation rollouts.

### Quality target

TODO: final

The quality target is pending MLCommons ratification. The current launcher reads `MLPERF_TARGET_ACCURACY` and defaults to `1.0`.

### Evaluation frequency

| Parameter | Value |
| :-- | :-- |
| Evaluate at start | Yes |
| Evaluation period | Every 2 training steps |
| Evaluate at end | Yes |
| Maximum training steps | 20 |

### Evaluation thoroughness

The validation JSONL contains 256 R2E-Gym tasks and each evaluation uses the full validation set.
