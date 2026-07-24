# 1. Summary

## Plain-language summary

This benchmark measures how quickly a large language model learns to repair
real software projects. For each task, the model receives a problem report and
a copy of the affected project. It can inspect files, edit code, and run
commands in an isolated environment. It receives credit only when the final
change passes the task's automated evaluation.

Every run starts from the same pretrained model and trains on the same ordered
set of software tasks. The measured result is the time needed to reach a fixed
success rate on a separate set of tasks that the model did not train on.

## Technical summary

This benchmark uses Reinforcement Learning with Verifiable Rewards (RLVR) to
post-train [`Qwen/Qwen3.5-397B-A17B`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
with Group Relative Policy Optimization (GRPO). The policy acts through a
CodeAct/OpenHands software-engineering agent in NeMo Gym. Each task provides a
source repository and an issue description; the agent edits the repository in
an isolated R2E-Gym environment. A rollout receives reward 1 when the generated
patch passes the task evaluation and reward 0 otherwise.

The objective is to reach the target validation `pass@4` on a held-out set of
R2E-Gym tasks. Training and generation use disaggregated GPU pools coordinated
through Ray:

- Megatron Bridge and Megatron Core train the policy.
- vLLM serves generation and receives updated policy weights after every
  training step.
- NeMo Gym and OpenHands execute the multi-turn software-engineering rollouts
  in per-task Apptainer/Singularity containers.

The benchmark implementation is provided by
[NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL/tree/mlperf-training-qwen35-main).
This repository checks out NeMo-RL under the local submodule path
`llm_moe_grpo/RL`, pinned to commit
`3fca04c9b313d313302923a5bb6b0c8dc0340ed6`. The source commit, recursive
submodule revisions, Python dependency lock, and downstream build-time patches
are recorded in the benchmark container.

The principal entrypoints are:

| Purpose | Path |
|---|---|
| Dataset download and preprocessing | `download_dataset.sh` |
| Dataset identity verification | `verify_dataset.sh` |
| Pretrained-checkpoint download | `download_checkpoint.sh` |
| Pretrained-checkpoint conversion | `RL/docker/mlperf/data_scripts/convert_ckpt.sh` |
| Benchmark run and timing | `RL/docker/mlperf/run_and_time.sh` |
| Container recipe | `RL/docker/mlperf/Dockerfile` |
| RCP submission wrapper | `RL/docker/mlperf/submit_rcp.sh` |
| Slurm/Ray launcher | `RL/docker/mlperf/run.sub` |
| Common benchmark recipe | `RL/qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml` |
| GBS-specific recipes | `RL/qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_gbs{256,512,1024}.yaml` |
| Detailed replica launch procedure | [`qwen35_grpo_rcp_launch_instructions.md`](qwen35_grpo_rcp_launch_instructions.md) |

# 2. Directions

## System requirements

The qualified reference configuration targets 256 NVIDIA GB300 GPUs and uses
the supplied `linux/arm64` benchmark image. A deployment must provide:

- a distributed execution environment capable of running the NeMo-RL policy,
  generation, and environment workers across the target GPUs;
- a container mechanism for the benchmark image and isolated R2E-Gym task
  environments; and
- storage that makes the model checkpoint, converted checkpoint cache,
  training and validation data, task environments, and result directory
  available to the participating processes.

The supplied launcher encodes the qualified node layout, policy/generation
split, topology placement, and Slurm integration. Those are properties of the
reference implementation rather than independent infrastructure requirements.
Reference development and RCP qualification used NVIDIA GB300 GPUs in NVL72
systems, with the qualified job exposed to the launcher as 64 four-GPU
scheduler nodes.

## Initialize the reference source

From the root of this repository:

```bash
git submodule update --init --recursive llm_moe_grpo/RL
cd llm_moe_grpo/RL

test "$(git rev-parse HEAD)" = \
  "3fca04c9b313d313302923a5bb6b0c8dc0340ed6"
git submodule update --init --recursive
```

The immutable commit check is intentional. A container and launcher checkout
from different revisions are not an equivalent benchmark environment.

## Build the benchmark container

Build from `RL/docker/mlperf/Dockerfile`, not from the former NeMo-RL v0.6.0
Gym-overlay Dockerfile. The current recipe starts from a digest-pinned public
CUDA development image, checks out the requested NeMo-RL revision and its
submodules, installs the locked dependencies, applies the benchmark's guarded
patches, builds the Arm HybridEP dependency, installs Apptainer, and prefetches
the Ray and NeMo Gym virtual environments.

From the `llm_moe_grpo/RL` directory:

```bash
source_commit="$(git rev-parse HEAD)"
source_repo="$(git remote get-url origin)"
image="<registry>/<repository>/qwen35_397b_grpo:${source_commit:0:12}"

docker buildx build \
  --platform linux/arm64 \
  --build-arg NEMO_RL_REPO="${source_repo}" \
  --build-arg NEMO_RL_REVISION="${source_commit}" \
  --build-arg GIT_COMMIT_ID="${source_commit}" \
  --tag "${image}" \
  --push \
  --file docker/mlperf/Dockerfile \
  docker/mlperf
```

`NEMO_RL_REPO` is taken from the submodule checkout so the Docker build fetches
the same repository that supplies the pinned commit. Both
`NEMO_RL_REVISION` and `GIT_COMMIT_ID` must be immutable commit hashes, not
branch names.

The build does not use a NeMo-RL nightly image. Its default base is
`nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04` at the digest
pinned in the Dockerfile. The resulting image writes its provenance to
`/NEMO_RL_PROVENANCE.txt`.

The supplied launcher receives the benchmark image through `CONT`. Depending
on the site's container infrastructure, this may be a registry reference or a
runtime-specific local image. Image import, authentication, and storage are
installation-specific.

The authoritative run uses only source and recipes baked into this image. Do
not use a host source overlay or a runtime patch:

```bash
unset NRL_RUNTIME_PATCH NRL_RUNTIME_PATCH_CONTAINER REPO_LOCATION EXTRA_MOUNTS
export NRL_SOURCE_OVERLAY=0
```

## Prepare model and data artifacts

The run requires five external artifacts:

| Artifact | Runtime variable | Required identity or behavior |
|---|---|---|
| Qwen3.5 policy checkpoint | `HF_CKPT_PATH` | `Qwen/Qwen3.5-397B-A17B` snapshot `8472618112abcbd45acbcdc58436aff4233c23f7` |
| Megatron checkpoint cache | `NRL_MEGATRON_CHECKPOINT_DIR` | Writable directory; an empty directory is converted from the HF checkpoint on first use |
| Curriculum training JSONL | `QWEN35_CURRICULUM_DATA_PATH` | Exact 685-row curriculum-v2 artifact described below |
| Validation JSONL | `NEMO_GYM_SWE_VALIDATION_DATA_PATH` | 256 held-out R2E-Gym tasks |
| Per-task SIF images | `NEMO_GYM_SWE_SIF_DIR` | All 941 environments referenced by the qualified train/validation lists; the checked-in helper builds a compatible 977-image superset |

### Policy checkpoint

The checked-in download helper pins the exact Hugging Face model revision and
uses the standard Hugging Face cache layout:

```bash
cd llm_moe_grpo

export HF_HOME="<shared-path>/huggingface"
export HF_TOKEN="<read-token>"  # if required by the installation

./download_checkpoint.sh

export HF_CKPT_PATH="<shared-path>/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/8472618112abcbd45acbcdc58436aff4233c23f7"
mkdir -p "<shared-path>/qwen35-mcore-cache"
export NRL_MEGATRON_CHECKPOINT_DIR="<shared-path>/qwen35-mcore-cache"
```

`HF_CKPT_PATH` must name the snapshot directory, not only the Hugging Face
cache root. `config_mounts.sh` recognizes the `snapshots/<revision>` layout and
also mounts the corresponding `blobs` directory so the snapshot symlinks
resolve inside the container.

The policy starts from this pretrained checkpoint. It is not randomly
initialized. NeMo-RL converts it to the Megatron format expected by this
source revision and stores the result in
`NRL_MEGATRON_CHECKPOINT_DIR`. Reusing a Megatron cache produced by an
incompatible NeMo-RL/Megatron Bridge/Megatron Core stack is not supported.

### Training and validation JSONL

The benchmark training input is:

```text
benchmark_r2e_gym_easy_train.filtered.curriculum-v2-classic-cycles2-seed20260710.jsonl
```

Its known identity is:

| Property | Value |
|---|---:|
| Rows | 685 |
| Size | 444,421,669 bytes |
| SHA-256 | `c07bcd64ed1c558e28d091239104e38295a5e696c1d21bb0b61f0346c7eaa0f7` |

The checked-in converter and instance-ID lists reproduce this filtered,
curriculum-ordered input. Commit
`9e64cc37197c9d00954ee8285c623fd7e53595e2`, which is included in the pinned
NeMo-RL revision, removes 36 unusable environments from the former 721-row
training list and orders the remaining 685 rows deterministically.

The held-out validation input is:

```text
benchmark_r2e_gym_easy_val.jsonl
```

Its reference identity is:

| Property | Value |
|---|---:|
| Rows | 256 |
| Size | 173,801,096 bytes |
| SHA-256 | `452d0e6b3c1973669334062dc24931355de51749df1ab51fc9bb71a129f7bb5c` |

The package pins source dataset revision
`e8b9fcbce43eaca0dc2c0d4798ee6f3e965f590a` of
[`R2E-Gym/R2E-Gym-Subset`](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset).
The download script retrieves that revision, runs the pinned converter, gives
the curriculum output its canonical name, and invokes the independent
verification script. The preprocessing host requires `uv`/`uvx` and Git
network access for repository metadata not already present in `cache_dir`.

```bash
cd llm_moe_grpo

dataset_dir="<work-path>/R2E-Gym__R2E-Gym-Subset"
output_dir="<output-path>"
cache_dir="<work-path>/r2e-repository-cache"

./download_dataset.sh \
  --dataset-dir "${dataset_dir}" \
  --output-dir "${output_dir}" \
  --cache-dir "${cache_dir}"

./verify_dataset.sh "${output_dir}"
```

`download_dataset.sh` uses
`RL/tools/create_r2e_gym_easy_subset_jsonl.py`,
`RL/tools/train-instance-ids.txt`, and `RL/tools/val-instance-ids.txt`.
The similarly named converter under `RL/docker/mlperf/data_scripts/` retains
the earlier 721-row, non-curriculum training split and is not the qualified
data generator.

The curriculum algorithm is fixed in the script: seed `20260710`, 16 examples
per curriculum batch, and two easy-to-hard cycles. It ranks difficulty using
75% inverse classic pass rate and 25% changed-line count capped at 20, divides
the training set into 25% easy, 50% medium, and 25% hard buckets, and changes
the per-batch mixture through warmup, core, and hardening phases. The converter
also validates row structure, verifies that all requested IDs were found, and
rejects overlapping train and validation ID sets. The hash checks above are
the final identity gate because `data.shuffle: false` makes row order part of
the benchmark input.

### R2E-Gym task containers

The qualified lists under `RL/tools` contain 685 training IDs and 256
validation IDs, for 941 task environments used by the benchmark. The SIF
builder's committed manifest is a 977-environment superset: it also contains
the 36 training environments filtered out by the qualified training list.
Those extra images are never selected by the generated JSONL. The helper below
builds all 977 Arm Docker images, pushes them to a user-supplied registry, and
converts them to SIF:

```bash
cd llm_moe_grpo/RL

export DOCKER_REGISTRY="<registry>"
export DOCKER_USER="<registry-user>"
export DOCKER_TOKEN="<registry-token>"
export SIF_DIR="<shared-path>/r2e-gym-sif"
export WORK_DIR="<fast-local-work-path>"
export STATE_DIR="<persistent-build-state-path>"
export MAX_WORKERS="<parallel-build-count>"

bash docker/mlperf/data_scripts/build_r2e_gym_sif.sh
```

The build host must provide Docker and enough local capacity for concurrent
image builds. The helper mounts the Docker daemon socket into its
dataset-processing container. The recipe accepts either of these layouts:

```text
<NEMO_GYM_SWE_SIF_DIR>/r2egym/<instance_id>.sif
<NEMO_GYM_SWE_SIF_DIR>/<instance_id>.sif
```

## Create an external site data configuration

Keep installation paths, Slurm transport settings, and system metadata outside
the repository. Create a shell file containing:

```bash
export HF_CKPT_PATH="<HF_397B_SNAPSHOT>"
export NRL_MEGATRON_CHECKPOINT_DIR="<WRITABLE_MCORE_CHECKPOINT_CACHE>"
export NEMO_GYM_SWE_SIF_DIR="<R2E_GYM_SIF_DIRECTORY>"
export NEMO_GYM_SWE_VALIDATION_DATA_PATH="<VALIDATION_JSONL>"
export QWEN35_CURRICULUM_DATA_PATH="<CURRICULUM_V2_JSONL>"

export MLPERF_SUBMITTER="<ORGANIZATION>"
export MLPERF_STATUS="<SYSTEM_STATUS>"
export MLPERF_SYSTEM_NAME="<SYSTEM_NAME>"
export MLPERF_ACCELERATOR_MODEL_NAME="<ACCELERATOR_MODEL>"
export MLPERF_FRAMEWORK="NVIDIA NeMo RL"
export MLPERF_FRAMEWORK_SHORT_NAME="nemo_rl"
export MLPERF_HOST_STORAGE_TYPE="<STORAGE_TYPE>"
export MLPERF_HOST_STORAGE_CAPACITY="<STORAGE_CAPACITY>"
export MLPERF_HOST_NETWORKING="<HOST_NETWORK>"
export MLPERF_ACCELERATOR_MEMORY_CONFIGURATION="<ACCELERATOR_MEMORY>"
export MLPERF_ACCELERATOR_INTERCONNECT="<ACCELERATOR_INTERCONNECT>"
```

The system profile maps `QWEN35_CURRICULUM_DATA_PATH` to
`NEMO_GYM_SWE_TRAIN_DATA_PATH`. Cluster-specific communication-library or
transport variables may also be exported from this external file when the
installation requires them.

## Run and time the benchmark

Run the submission wrapper from the `llm_moe_grpo/RL` directory. Supply the
container, external data configuration, shared result root, and Slurm routing:

```bash
export CONT="<REGISTRY_IMAGE_OR_SQUASHFS>"
export RCP_DATA_CONFIG="<EXTERNAL_SITE_DATA_CONFIG>"
export RESULT_ROOT="<SHARED_RESULT_ROOT>"
export SLURM_PARTITION_NAME="<SLURM_PARTITION>"
export SLURM_ACCOUNT_NAME="<SLURM_ACCOUNT>"
```

The target-0.69 RCP configurations are:

| GBS | Prompts x generations | Learning rate and minimum LR | Gradient clip | First validation | Maximum steps |
|---:|---:|---:|---:|---:|---:|
| 256 | 16 x 16 | `1.0e-6` | `0.125` | 18 | 30 |
| 512 | 32 x 16 | `1.4142135624e-6` | `0.08838834765` | 10 | 20 |
| 1024 | 64 x 16 | `2.0e-6` | `0.0625` | 7 | 10 |

The learning-rate and clipping schedules represented by this table are:

```text
learning_rate(GBS) = 1.0e-6 * sqrt(GBS / 256)
gradient_clip(GBS) = 0.125 * sqrt(256 / GBS) = 2 / sqrt(GBS)
```

These are the benchmark's selected GBS-dependent settings, not a general GRPO
scaling rule.

### Reference hyperparameter controls

The qualified reference fixes the optimizer to Adam and does not allow an
alternative optimizer. The RCP wrapper exposes the benchmark-affecting study
controls as named command-line arguments with type checks:

| Control | Command-line argument | Qualified value or rule |
|---|---|---|
| Global batch size | `--gbs` | One of 256, 512, or 1024 for the checked-in RCPs |
| Learning rate and minimum LR | `--lr` | Exact GBS-dependent value in the table above |
| Maximum gradient norm | `--clip` | Exact GBS-dependent value in the table above |
| First validation step | `--val-start` | Exact GBS-dependent value in the table above |
| Maximum training steps | `--max-steps` | Exact GBS-dependent value in the table above |
| Quality target | `--target` | `0.69` |
| Replica count | `--replicas` | Positive integer |
| RNG seed | `--seed-base` | Nonnegative integer; omitted selects a random 32-bit seed |

All remaining optimizer, GRPO, generation, and agent hyperparameters are fixed
in the checked-in common and GBS-specific YAML recipes. The final list of
submitter-tunable parameters and legal alternative values is governed by the
MLPerf Training rules; the broader study controls accepted by
`submit_rcp.sh` do not by themselves authorize an alternative submission
configuration.

The first-validation values in the table are explicit submission arguments.
They intentionally override the child YAML defaults. Validation then runs at
every training step because the common recipe sets `grpo.val_period: 1`.

Submit one replica into an isolated result directory as follows:

```bash
gbs=256
replica=1

export LOGDIR="${RESULT_ROOT}/GBS${gbs}/r${replica}"
export SBATCH_OUTPUT="${LOGDIR}/slurm-%j.out"
mkdir -p "${LOGDIR}"

docker/mlperf/submit_rcp.sh \
  --data-config "${RCP_DATA_CONFIG}" \
  --gbs "${gbs}" \
  --val-start 18 \
  --max-steps 30 \
  --lr 1.0e-6 \
  --clip 0.125 \
  --target 0.69 \
  --replicas 1 \
  --name "qwen35-rcp-gbs${gbs}-rep${replica}" \
  --time 240 \
  --partition "${SLURM_PARTITION_NAME}" \
  --account "${SLURM_ACCOUNT_NAME}"
```

Run six independent replicas per GBS, changing the values according to the
configuration table and giving every replica a separate `LOGDIR`. The complete
three-GBS loop is provided in
[`qwen35_grpo_rcp_launch_instructions.md`](qwen35_grpo_rcp_launch_instructions.md).

If `--seed-base` is omitted, `submit_rcp.sh` obtains a nonnegative 32-bit seed
from `/dev/urandom`. Pass `--seed-base <NONNEGATIVE_INTEGER>` to reproduce an
individual run. When a single invocation submits multiple replicas, replica
seeds increment from that base; the isolated-directory procedure invokes the
wrapper once per replica, so explicitly selected seeds must be distinct.

Use `--dry-run` to print the resulting `sbatch` command without submitting.
The resolved reference allocation is 64 nodes x 4 GPUs, split into 16 policy
and 48 generation nodes, with a 240-minute Slurm and driver wall time.

`run.sub` starts one Ray worker unit per GPU, starts the GRPO driver once on
the Ray head node, and writes the MLPerf log under the mounted result
directory. The benchmark source remains baked into the image; the host
checkout is used only to submit the job.

# 3. Dataset and environment

## Publication and attribution

The task data originates from
[`R2E-Gym/R2E-Gym-Subset`](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset).
The fixed easy split retains tasks whose environments can be built for the
target architecture and whose source changes satisfy the easy-subset
selection. The qualified instance-ID files under `RL/tools` define disjoint
685-task training and 256-task validation sets.

The benchmark trains on the fixed 685-row curriculum-v2 artifact identified in
Section 2. It validates on all 256 held-out tasks. Runtime data shuffling is
disabled, so training row order is preserved.

## NeMo Gym and OpenHands configuration

The common recipe starts these NeMo Gym configurations:

```text
responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
responses_api_agents/swe_agents/configs/swebench_openhands_training.yaml
```

The generated JSONL selects the Gym agent definitions through
`agent_ref.type=responses_api_agents`: training rows use `swe_agents_train`,
and validation rows use `swe_agents_val`. In the pinned Gym configuration,
both definitions select `CodeActAgent`, set `diversify_tool_names: false`, and
use the same prompt templates:

| Prompt | Path in the `RL` checkout | SHA-256 |
|---|---|---|
| System | `3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/prompts/openhands/system_prompt.j2` | `8aff26bf83a11605ca9bab0336804d56d202f1555b0a073d21db21e4e41b8b9a` |
| User | `3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/prompts/openhands/user_prompt.j2` | `0b0489b934d2c229c439b7af2f588fe404a6327be9da3c12d4c7c9efd3cf1c03` |

The files and agent definitions above come from the pinned NeMo Gym revision
`610a08ab5fe9f8f5fb5fff36b170429ea67f0f92`. The agent implementation is
backed by the OpenHands revision
`0d766ad06b2be64a42e6f0175b9ebcc4a06599d9` selected by the common recipe.

These templates are benchmark inputs and must be used byte-for-byte when
reproducing the reference RCPs. Do not replace them with package defaults,
host-local templates, or custom prompt overrides. The user template injects
the task's repository path and issue description and instructs the agent to
make minimal non-test source changes. The system template defines the
OpenHands role, tool-use workflow, code-editing behavior, and safety
constraints. Replacing either template changes the agent policy presented to
the model and may change benchmark results.

Relevant environment settings are:

| Setting | Value |
|---|---:|
| OpenHands agent class | `CodeActAgent` |
| Maximum agent turns | 30 |
| Training concurrency | 256 |
| Validation concurrency | 256 |
| Training agent timeout | 1,800 seconds |
| Validation agent timeout | 1,800 seconds |
| Training test timeout | 300 seconds |
| Validation test timeout | 180 seconds |
| Task runtime | Apptainer/Singularity SIF |
| Model reasoning parser | `qwen3` |
| Tool parser | `qwen3_xml` |
| Qwen thinking mode | Disabled with `enable_thinking: false` |
| Sequential reasoning | Disabled |

Every training and validation rollout must reach a normal terminal result. A
rollout truncated by the agent timeout is incomplete and must not be included
as a valid benchmark sample. Configure both agent timeouts high enough that no
rollout is terminated by the timeout. The reference RCPs used 1,800 seconds for
both training and validation; this is a reference value, not an upper bound,
and should be increased when required to ensure rollout completion.

Training generation uses temperature `1.0`, top-p `1.0`, and no top-k
restriction. Validation generation uses temperature `0.1` and top-p `0.95`.
The benchmark context and maximum generation length are both capped at 65,536
tokens.

# 4. Model

## Publication and attribution

The initial policy is the Apache-2.0
[`Qwen/Qwen3.5-397B-A17B`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
checkpoint released by the Qwen team. The authoritative public architecture
sources are the
[`Qwen3.5-397B-A17B` model card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
and the Qwen team's
[`Qwen3.5: Towards Native Multimodal Agents`](https://qwen.ai/blog?id=qwen3.5)
release publication.

## Model details

| Config | Value |
|---|---:|
| Total parameters | 397B |
| Active parameters | 17B |
| Layers | 60 |
| Hidden layout | 15 blocks of 3 Gated DeltaNet + MoE layers followed by 1 gated-attention + MoE layer |
| Model dimension | 4,096 |
| Gated DeltaNet heads | 64 V / 16 QK |
| Gated DeltaNet head dimension | 128 |
| Gated-attention heads | 32 Q / 2 KV |
| Gated-attention head dimension | 256 |
| RoPE dimension | 64 |
| Routed experts | 512 |
| Active routed experts | 10 |
| Shared experts | 1 |
| Expert intermediate dimension | 1,024 |
| Vocabulary size | 248,320 |
| Native context length | 262,144 |
| Benchmark context length | 65,536 |

The benchmark uses the language-model path only; image inputs are not part of
the R2E-Gym task format. Qwen's MTP layers are not trained or used for
speculative decoding in this recipe (`mtp_num_layers: 0` and
`policy.draft.enabled: false`).

## Benchmark runtime

| Component | Reference RCP configuration |
|---|---|
| Policy training | 16 nodes x 4 GPUs; Megatron TP4, PP2, EP32, CP1; sequence parallelism and packed sequences enabled; BF16 policy precision |
| Policy attention | FlashAttention backend through Transformer Engine, including the container's guarded SM103 allowlist update |
| MoE dispatch | Megatron `flex` token dispatcher with the HybridEP backend and 32 SMs |
| Generation | 48 nodes x 4 GPUs; vLLM TP8, PP1, EP8; non-colocated with policy training |
| vLLM MoE kernels | Triton backend, preserving the refit-compatible weight layout |
| vLLM scheduling | Chunked prefill, 16,384 maximum batched tokens, throughput performance mode |
| vLLM CUDA graphs | `VLLM_COMPILE` with `FULL_AND_PIECEWISE` graph mode |
| Prefix caching | Enabled and invalidated after policy refits |
| Policy refit | Updated policy weights are transferred to vLLM after each training step |

The pinned runtime identities are:

| Component | Version | Revision or source |
|---|---|---|
| NeMo-RL | source reports `0.6.0` | `3fca04c9b313d313302923a5bb6b0c8dc0340ed6` |
| NeMo Gym | `0.4.0rc0` | `610a08ab5fe9f8f5fb5fff36b170429ea67f0f92` |
| Megatron Bridge | `0.6.0` | `554c7b9324225aa863eee52e8b8fdde7abced2b1` |
| Megatron Core | `0.19.0` | `002255075c3728fded9a2e435677840b08560d55` |
| Automodel | editable source | `24b47e856263d313b942f0ed666c63fff83306b4` |
| vLLM | `0.20.0` | locked Arm wheel installed by the benchmark Dockerfile |
| Transformer Engine | `2.15.0+42b8400` | `42b840051647eef89761a16dfdff87e82bb253ab` |
| Ray | `2.55.1` | Python lockfile |
| HybridEP for policy workers | upstream DeepEP checkout | `e0a5b1d9848ab3e7b4a67842bf06f067bfac67f8` |
| OpenHands agent framework | pinned source | `0d766ad06b2be64a42e6f0175b9ebcc4a06599d9` |
| R2E-Gym evaluation harness | pinned source | `6823e64f94ae645f5265c03af0eb2e8523530a0d` |

The immutable runtime authority is the built image's
`/NEMO_RL_PROVENANCE.txt`, which also records the base-image digest, recursive
submodule status, lockfile hash, dependency records, and downstream patch
hashes.

The checked-in RCPs were generated with policy and generation math in BF16,
while optimizer parameters and the MoE router used FP32. These values document
the reference runs; they do not prescribe the precision used by other
submissions. Permitted submission precision is governed by the applicable
MLPerf Training rules.

## Weight initialization

Training starts from the pretrained Hugging Face policy converted to
Megatron-Core format. In the checked-in RCPs, MoE router weights are frozen
(`freeze_moe_router: true`), policy weights remain BF16, and optimizer
parameters are FP32.

## Loss function

The recipe uses token-level GRPO with:

- binary environment rewards;
- reward normalization;
- a leave-one-out baseline;
- no reward shaping;
- no reference-policy KL penalty (`reference_policy_kl_penalty: 0`);
- asymmetric PPO ratio clipping configured as `0.2` and `0.28`;
- asynchronous collection with a maximum trajectory age of one training step;
- sequence-mask truncated importance sampling (`seq-mask-tis`) with lower and
  upper bounds `0.999` and `1.002`;
- importance-sampling correction enabled;
- `force_on_policy_ratio: true`;
- reference-policy log-probability calculation skipped; and
- overlong-response filtering enabled.

The `[0.999, 1.002]` bounds replace the wider bounds used in earlier
development configurations.

## Optimizer

The Megatron recipe selects `optimizer: adam`. The MLPerf logger records this
under the ruleset's `adamw` optimizer name; configured weight decay is zero.
The optimizer uses distributed, precision-aware state.

| Parameter | Reference RCP value |
|---|---|
| Optimizer | Adam (`optimizer: adam`; MLPerf log name `adamw`) |
| Base learning rate | `1.0e-6 * sqrt(GBS / 256)` |
| End/minimum learning rate | Equal to the base learning rate |
| Learning-rate schedule | Constant |
| Warmup steps | 0 |
| Weight decay | `0.0` |
| Adam beta1 | `0.9` |
| Adam beta2 | `0.999` |
| Adam epsilon | `1e-8` |
| Gradient clipping | `0.125 * sqrt(256 / GBS)` |
| Distributed optimizer | Enabled |
| Precision-aware optimizer | Enabled |
| Optimizer parameter dtype | FP32 |
| Policy precision | BF16 |

The pinned NeMo-RL implementation supports writing and resuming full policy and
optimizer checkpoints. Periodic checkpointing and optimizer-state saving are
disabled in the RCP time-to-target recipe so checkpoint I/O is not part of the
measured run.

# 5. Quality

## Quality metric

The quality metric is observed grouped `pass@4` on the 256-task validation
set. Each task is evaluated with four independently sampled trajectories. A
task passes when at least one of its four rewards is positive:

```text
pass@4 = mean_over_tasks(any(reward[task, generation] > 0))
```

One complete validation therefore executes 1,024 agent trajectories. NeMo-RL
also reports all-four-pass `pass^4` and average pass@1 across the four
trajectories as diagnostics, but the MLPerf `eval_accuracy` and convergence
decision use `pass@4`.

## Quality target

The quality target is:

```text
pass@4 >= 0.69
```

On a 256-task validation set, the first representable score meeting this
threshold is `177 / 256 = 0.69140625`: at least 177 held-out tasks must be
solved by one or more of their four attempts.

The checked-in RCP logs provide six independent seeds at each qualified GBS:

| GBS | RCP logs | Runs reaching target | First observed crossing | Observed pass@4 range at crossing |
|---:|---|---:|---:|---:|
| 256 | `rcp_logs/256/seed_*.out` | 6 / 6 | Step 18 | 0.703125 - 0.75390625 |
| 512 | `rcp_logs/512/seed_*.out` | 6 / 6 | Step 10 | 0.6953125 - 0.73046875 |
| 1024 | `rcp_logs/1024/seed_*.out` | 6 / 6 | Step 7 | 0.70703125 - 0.75000000 |

All 18 runs therefore reach the target at the first scheduled validation.
This is stable observed convergence at the configured evaluation schedule; it
does not imply zero latent crossing-step variance before the first
observation.

When an evaluation reaches the target, the MLPerf logger emits a successful
`run_stop` and terminates training. The submission wrapper must receive
`--target 0.69`; an empty target disables target-based stopping and is not the
qualified time-to-target run.

## Evaluation schedule

| GBS | Evaluate at start | First evaluation | Period after first evaluation | Evaluate separately at end | Maximum steps |
|---:|---|---:|---:|---|---:|
| 256 | No | 18 | Every step | No | 30 |
| 512 | No | 10 | Every step | No | 20 |
| 1024 | No | 7 | Every step | No | 10 |

The validation dataloader uses the entire 256-task validation JSONL at every
evaluation. The four trajectories for each task use validation temperature
`0.1` and top-p `0.95`.

Each validation executes 1,024 complete agent trajectories, so evaluation is
substantially more expensive than a conventional forward-only validation
pass. The schedule delays the first evaluation to the GBS-specific RCP
convergence window, then evaluates every step so a later crossing is detected
within one additional training step.
