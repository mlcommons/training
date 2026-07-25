# Qwen 3.5 target-0.69 RCP launch instructions

These instructions submit six independent replicas for each Qwen 3.5 RCP
configuration. Storage locations, Slurm partition and account names, and
MLPerf system metadata are supplied at runtime.

## Prerequisites

Use a source checkout and container built from the same revision. The checkout
must contain:

```text
docker/mlperf/submit_rcp.sh
docker/mlperf/run.sub
docker/mlperf/config_GB300_64x4_t16g48_tp4pp2ep32gtp8.sh
qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml
qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_gbs256.yaml
qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_gbs512.yaml
qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_gbs1024.yaml
```

Run from the repository root.

The profile uses the source and recipes baked into the container. Do not set a
source overlay or runtime patch:

```bash
unset NRL_RUNTIME_PATCH NRL_RUNTIME_PATCH_CONTAINER REPO_LOCATION EXTRA_MOUNTS
export NRL_SOURCE_OVERLAY=0
```

## Data configuration

Create a shell file outside the repository containing the paths and MLPerf
metadata for the target installation:

```bash
export HF_CKPT_PATH="<HF_397B_CHECKPOINT>"
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

The selected system profile maps `QWEN35_CURRICULUM_DATA_PATH` to
`NEMO_GYM_SWE_TRAIN_DATA_PATH`.

Set any cluster transport or communication-library environment in this
external file when required by the target installation. Those settings are
not part of the training recipe.

## Configurations

| GBS | Child recipe | Prompts x generations | First validation | Maximum steps | LR and minimum LR | Gradient clip |
|---:|---|---:|---:|---:|---:|---:|
| 256 | `grpo_qwen35_397b_swe_openhands_async_gbs256.yaml` | 16 x 16 | 18 | 30 | `1.0e-6` | `0.125` |
| 512 | `grpo_qwen35_397b_swe_openhands_async_gbs512.yaml` | 32 x 16 | 10 | 20 | `1.4142135624e-6` | `0.08838834765` |
| 1024 | `grpo_qwen35_397b_swe_openhands_async_gbs1024.yaml` | 64 x 16 | 7 | 10 | `2.0e-6` | `0.0625` |

Validation runs every step after the configured first validation because the
common recipe sets `grpo.val_period=1`.

## Submission

Supply the runtime-specific values:

```bash
export CONT="<CONTAINER_IMAGE_OR_SQUASHFS>"
export RCP_DATA_CONFIG="<EXTERNAL_DATA_CONFIG>"
export RESULT_ROOT="<RESULT_ROOT>"
export SLURM_PARTITION_NAME="<SLURM_PARTITION>"
export SLURM_ACCOUNT_NAME="<SLURM_ACCOUNT>"
```

Use one `submit_rcp.sh` invocation per replica so each job has an isolated
result directory:

```bash
submit_one() {
    local gbs="$1"
    local replica="$2"
    local val_start="$3"
    local max_steps="$4"
    local learning_rate="$5"
    local max_grad_norm="$6"

    export LOGDIR="${RESULT_ROOT}/GBS${gbs}/r${replica}"
    export SBATCH_OUTPUT="${LOGDIR}/slurm-%j.out"
    mkdir -p "${LOGDIR}"

    docker/mlperf/submit_rcp.sh \
        --data-config "${RCP_DATA_CONFIG}" \
        --gbs "${gbs}" \
        --val-start "${val_start}" \
        --max-steps "${max_steps}" \
        --lr "${learning_rate}" \
        --clip "${max_grad_norm}" \
        --target 0.69 \
        --replicas 1 \
        --name "qwen35-rcp-gbs${gbs}-rep${replica}" \
        --time 240 \
        --partition "${SLURM_PARTITION_NAME}" \
        --account "${SLURM_ACCOUNT_NAME}"
}

for replica in $(seq 1 6); do
    submit_one 256 "${replica}" 18 30 1.0e-6 0.125
done

for replica in $(seq 1 6); do
    submit_one 512 "${replica}" 10 20 1.4142135624e-6 0.08838834765
done

for replica in $(seq 1 6); do
    submit_one 1024 "${replica}" 7 10 2.0e-6 0.0625
done
```

### Seed selection

If `--seed-base` is omitted, `submit_rcp.sh` generates a nonnegative
32-bit seed from `/dev/urandom` for the submitted job.

For a reproducible run, add an explicit seed to the `submit_rcp.sh`
invocation:

```bash
--seed-base "<NONNEGATIVE_INTEGER>"
```

When `--replicas 1` is used, that value is the exact `grpo.seed` for the job.
When one invocation uses `--replicas N`, replica 1 uses the base value and each
later replica increments it by one. The isolated-directory loop above uses
one invocation per replica, so supply a distinct `--seed-base` value to each
call if exact seeds must be predetermined.

Use `--dry-run` on the `submit_rcp.sh` command to inspect the resulting
`sbatch` invocation without submitting.

## Resolved launch shape

The checked-in profile resolves every job to:

```text
64 nodes
4 GPUs per node
16 policy nodes
48 generation nodes
segment size 8
non-colocated generation
240-minute Slurm and driver wall time
```

The effective Slurm command has this shape:

```text
sbatch --export=ALL \
  -N 64 \
  --time=240 \
  --segment=8 \
  --gres=gpu:4 \
  --partition=<SLURM_PARTITION> \
  --account=<SLURM_ACCOUNT> \
  --job-name=<EXPERIMENT_NAME> \
  docker/mlperf/run.sub
```

## Configuration flow

The launch path is:

```text
docker/mlperf/submit_rcp.sh
  -> source the external data configuration
  -> select qwen_35/configs/..._gbs<GBS>.yaml
  -> source config_GB300_64x4_t16g48_tp4pp2ep32gtp8.sh
  -> sbatch docker/mlperf/run.sub
       -> create the 64-node Ray cluster
       -> start the driver on the Ray head
       -> /workspace/llm/run_and_time.sh
            -> uv run /workspace/llm/run_grpo_nemo_gym.py
```

`run_and_time.sh` translates the named submitter arguments into these Hydra
overrides:

```text
++logger.mlperf_enabled=True
++logger.mlperf.log_file=/results/<DATESTAMP>_1_mllog.log
++logger.mlperf.benchmark=qwen35_397b_grpo
++logger.mlperf.target_accuracy=0.69
++logger.mlperf.force_success_status=False

++cluster.num_nodes=64
++cluster.gpus_per_node=4
++cluster.segment_size=8

++policy.generation.colocated.resources.num_nodes=48
++policy.generation.colocated.resources.gpus_per_node=4

++logger.wandb.name=<EXPERIMENT_NAME>
++logger.log_dir=/logs
++checkpointing.checkpoint_dir=/checkpoint/1

++grpo.seed=<REPLICA_SEED>
++grpo.max_num_steps=<MAX_STEPS>
++grpo.val_start_at=<FIRST_VALIDATION_STEP>

++policy.megatron_cfg.optimizer.lr=<LEARNING_RATE>
++policy.megatron_cfg.optimizer.min_lr=<LEARNING_RATE>
++policy.max_grad_norm=<MAX_GRAD_NORM>
```

The selected child recipe supplies `grpo.num_prompts_per_step` and
`grpo.num_generations_per_prompt`.

## Shared recipe behavior

The child recipes inherit
`qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml`. Relevant shared
settings are:

```text
validation:
  period: every step after val_start_at
  val_at_start: false
  val_at_end: false
  generations per prompt: 4
  temperature: 0.1
  top_p: 0.95

training generation:
  generations per prompt: 16
  temperature: 1.0
  top_p: 1.0
  max model/sequence length: 65536

GRPO:
  async enabled: true
  maximum trajectory age: 1 (reference RCP setting; not constrained)
  sequence-mask TIS: [0.999, 1.002]
  sequence logprob-error mask threshold: 2.0
  force_on_policy_ratio: true
  reference logprobs skipped: true
  checkpointing: disabled

policy:
  16 nodes x 4 GPUs
  TP4, PP2, EP32, CP1
  sequence parallelism: true
  packed sequences: true
  FlashAttention backend
  flex MoE dispatcher with HybridEP backend
  MTP training: disabled and not permitted

generation:
  48 nodes x 4 GPUs
  vLLM TP8, PP1, EP8
  speculative decoding: disabled and not permitted
  train and validation concurrency: 256
  train and validation agent timeout: 1800 seconds
  prefix caching: enabled
  chunked prefill: enabled
  max_num_batched_tokens: 16384
  performance_mode: throughput
  compilation mode: VLLM_COMPILE
  CUDA graph mode: FULL_AND_PIECEWISE
  MoE backend: triton
```

## Mount contract

`docker/mlperf/config_mounts.sh` maps runtime paths to stable container paths:

```text
<RESULT_DIR>                 -> /results
<RESULT_DIR>/nemo_logs_*     -> /logs
<RESULT_DIR>/checkpoint      -> /checkpoint
<RESULT_DIR>/hf_cache        -> /opt/nemo-rl/.cache
<HF_CHECKPOINT>              -> identity-mounted host path
<MCORE_CACHE>                -> /inputs/nemo_gym/mcore_ckpt
<TRAIN_JSONL>                -> /inputs/nemo_gym/data/train.jsonl, read-only
<VALIDATION_JSONL>           -> /inputs/nemo_gym/data/validation.jsonl, read-only
<SIF_DIRECTORY>              -> /inputs/nemo_gym/sif, read-only
/dev/fuse                    -> /dev/fuse
```

The source checkout is not mounted into `/opt/nemo-rl`.
