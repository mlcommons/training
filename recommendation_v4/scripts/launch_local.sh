#!/bin/bash
# =============================================================================
# launch_local.sh — single-host, NON-SLURM launcher for the yambda-5b trainer.
#
# This is the SLURM-free analog of scripts/launch_slurm.sh's `worker` phase:
# it sets the single-node distributed topology + sane env and invokes the SAME
# entry point (`train_ranker.py --dataset yambda-5b`) reading the SAME
# train/gin/yambda_5b.gin config. No scheduler, no docker, no RDMA overlay —
# everything runs directly on this host against an already-prepared dataset.
#
# Use it to:
#   * Smoke-test the launch path on a single GPU box (SMOKE=1, the default —
#     a few train/eval batches of one streaming window), or
#   * Run the full gin-default workload (SMOKE=0 — consumes whole windows).
#
# PREREQUISITES
#   1) Data prepared (run once, CPU-only — no GPU needed):
#        python generative_recommenders/dlrm_v3/preprocess_public_data.py \
#            --dataset yambda-5b --data-path "$DLRM_DATA_PATH"
#      producing  $DLRM_DATA_PATH/processed_5b/{train_sessions.parquet,...}
#             and $DLRM_DATA_PATH/shared_metadata/{artist,album}_item_mapping.parquet
#   2) The train_recipe GPU stack importable by $PYTHON (see docs/training_recipe.md):
#        torch (rocm or cuda build), fbgemm_gpu, torchrec, polars-u64-idx,
#        gin-config, xxhash, pandas, tensorboard, ...
#      This box must have visible GPUs (the trainer shards embeddings onto HBM).
#
# USAGE
#   # smoke (default): one window, 20 train + 10 eval batches
#   DLRM_DATA_PATH=/home/chcai/dlrm_data bash scripts/launch_local.sh
#
#   # full gin-default run (whole windows; long)
#   SMOKE=0 DLRM_DATA_PATH=/home/chcai/dlrm_data bash scripts/launch_local.sh
#
#   # restrict to 2 GPUs, custom log, plain (non-streaming) train-eval
#   GPUS_PER_NODE=2 MODE=train-eval LOG=/tmp/y.log bash scripts/launch_local.sh
#
# Every knob below is env-overridable; defaults reproduce launch_slurm.sh's
# single-node smoke path so a local run matches the known-good cluster path.
# =============================================================================
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

# ---- interpreter ------------------------------------------------------------
# Default to the venv created for data prep if present, else system python3.
# Override with PYTHON=/path/to/python (e.g. the in-container recipe python).
DEFAULT_PY=/home/chcai/dlrmv4_venv/bin/python
PYTHON=${PYTHON:-$([ -x "$DEFAULT_PY" ] && echo "$DEFAULT_PY" || echo python3)}

# ---- dataset / data path ----------------------------------------------------
DATASET=${DATASET:-yambda-5b}
MODE=${MODE:-streaming-train-eval}
# Mirrors the yambda_5b.gin default ("/apps/chcai/dlrm_data"); point at wherever
# preprocess_public_data.py wrote processed_5b/ + shared_metadata/.
export DLRM_DATA_PATH=${DLRM_DATA_PATH:-/home/chcai/dlrm_data}

LOG=${LOG:-$REPO_ROOT/yambda_local.$(date +%Y%m%d_%H%M%S).log}

# ---- single-node distributed topology --------------------------------------
# train_ranker reads these from the env (see train_ranker.main): it spawns
# GPUS_PER_NODE ranks via torch.multiprocessing on THIS host. localhost
# rendezvous; empty MASTER_PORT => train_ranker picks a free port.
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-}
# GPUS_PER_NODE: 0/unset => train_ranker auto-detects torch.cuda.device_count().
export GPUS_PER_NODE=${GPUS_PER_NODE:-0}

# ---- runtime env (matches launch_slurm.sh worker defaults) ------------------
export HSTU_HAMMER_KERNEL=${HSTU_HAMMER_KERNEL:-TRITON}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
# Single-node RCCL bootstrap: all ranks rendezvous over localhost, so pin the
# loopback NIC. Left to auto-detect, RCCL can grab a non-routable per-GPU RoCE
# NIC and hang/"No route to host" at init (same failure launch_slurm.sh pins
# fenic0 to avoid). Override NCCL_SOCKET_IFNAME for a routable multi-host setup.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# ---- smoke caps -------------------------------------------------------------
# SMOKE=1 (default): apply small per-window batch caps so a launch finishes in
# minutes (validates the path end-to-end). SMOKE=0: leave the gin defaults
# untouched (consume full windows — the real workload).
SMOKE=${SMOKE:-1}
if [ "$SMOKE" = "1" ]; then
  export START_TS=${START_TS:-150}
  export NUM_TRAIN_TS=${NUM_TRAIN_TS:-1}
  export NUM_TRAIN_BATCHES=${NUM_TRAIN_BATCHES:-20}
  export NUM_EVAL_BATCHES=${NUM_EVAL_BATCHES:-10}
  export EVAL_EVERY_N_WINDOWS=${EVAL_EVERY_N_WINDOWS:-1}
  export METRIC_LOG_FREQ=${METRIC_LOG_FREQ:-5}
  # Smaller per-sample shape keeps the smoke run light; drop these to use the
  # gin defaults (4086/4096). Reuse an existing hstu_cache_L<N>/ if present.
  export BATCH_SIZE=${BATCH_SIZE:-32}
fi

mkdir -p "$(dirname "$LOG")"
{
  echo "[$(date)] launch_local: dataset=$DATASET mode=$MODE smoke=$SMOKE"
  echo "[$(date)] PYTHON=$PYTHON"
  echo "[$(date)] DLRM_DATA_PATH=$DLRM_DATA_PATH"
  echo "[$(date)] topology: nnodes=$NNODES node_rank=$NODE_RANK gpus_per_node(req)=$GPUS_PER_NODE master=$MASTER_ADDR:${MASTER_PORT:-<auto>}"
} | tee -a "$LOG"

# ---- preflight: data present? ----------------------------------------------
SUFFIX=${DATASET#yambda-}
PROCESSED="$DLRM_DATA_PATH/processed_${SUFFIX}/train_sessions.parquet"
META="$DLRM_DATA_PATH/shared_metadata/artist_item_mapping.parquet"
if [ "$DATASET" = "yambda-5b" ] && { [ ! -f "$PROCESSED" ] || [ ! -f "$META" ]; }; then
  echo "[$(date)] ERROR: prepared data not found." | tee -a "$LOG"
  echo "  expected: $PROCESSED" | tee -a "$LOG"
  echo "        and: $META" | tee -a "$LOG"
  echo "  run preprocessing first:" | tee -a "$LOG"
  echo "    $PYTHON generative_recommenders/dlrm_v3/preprocess_public_data.py --dataset $DATASET --data-path $DLRM_DATA_PATH" | tee -a "$LOG"
  exit 1
fi

# ---- preflight: GPU stack importable + GPUs visible? ------------------------
echo "[$(date)] preflight: checking torch / fbgemm_gpu / torchrec + GPU count" | tee -a "$LOG"
"$PYTHON" - <<'PY' 2>&1 | tee -a "$LOG"
import sys
missing = []
for m in ("torch", "fbgemm_gpu", "torchrec", "polars", "gin", "xxhash"):
    try:
        __import__(m)
    except Exception as e:
        missing.append(f"{m} ({e.__class__.__name__})")
if missing:
    print("PREFLIGHT FAIL: missing/broken imports: " + ", ".join(missing))
    print("Install the train_recipe GPU stack (see docs/training_recipe.md).")
    sys.exit(3)
import torch
n = torch.cuda.device_count()
print(f"imports OK, torch {torch.__version__}, cuda/hip available={torch.cuda.is_available()}, {n} GPU(s)")
if n == 0:
    print("PREFLIGHT FAIL: no GPUs visible — the HSTU trainer shards embeddings "
          "onto GPU HBM and cannot run CPU-only. Launch on a GPU host.")
    sys.exit(4)
PY
pf=${PIPESTATUS[0]}
if [ "$pf" -ne 0 ]; then
  echo "[$(date)] preflight failed (rc=$pf) — not launching trainer." | tee -a "$LOG"
  exit "$pf"
fi

# ---- launch -----------------------------------------------------------------
echo "[$(date)] launching train_ranker ($DATASET, mode=$MODE)" | tee -a "$LOG"
"$PYTHON" -m generative_recommenders.dlrm_v3.train.train_ranker \
    --dataset "$DATASET" --mode "$MODE" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[$(date)] launch_local finished rc=$rc" | tee -a "$LOG"
exit "$rc"
