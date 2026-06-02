#!/bin/bash
# 8-GPU yambda-5b run. Resolves the package root from this script's location,
# so it works from any container mount point. Dataset path is in the gin file
# (generative_recommenders/dlrm_v3/train/gin/yambda_5b.gin).
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

LOG=${LOG:-/apps/chcai/yambda_5b_8gpu.log}
echo "[$(date)] REPO_ROOT=$REPO_ROOT" | tee "$LOG"

# polars-u64-idx (NOT stock polars) — yambda parquet's flat-explode overruns
# 32-bit row index. Reserved node has no outbound DNS, so we install from a
# pre-staged tarball under /apps/chcai/. Override PIP_LOCAL_TGZ for other hosts.
PIP_LOCAL_TGZ=${PIP_LOCAL_TGZ:-/apps/chcai/pip_local_yambda.tgz}
PIP_LOCAL_DIR=${PIP_LOCAL_DIR:-/tmp/pip_local}
if [ ! -f "$PIP_LOCAL_DIR/lib/python3.12/site-packages/polars/__init__.py" ]; then
  rm -rf "$PIP_LOCAL_DIR"
  mkdir -p "$PIP_LOCAL_DIR" && tar xzf "$PIP_LOCAL_TGZ" -C "$(dirname "$PIP_LOCAL_DIR")" 2>&1 | tail -3 | tee -a "$LOG"
fi

export PYTHONPATH="$PIP_LOCAL_DIR/lib/python3.12/site-packages:$REPO_ROOT:${PYTHONPATH:-}"
export HOME=${HOME:-/tmp}
echo "[$(date)] PYTHONPATH=$PYTHONPATH" | tee -a "$LOG"
python -c "import torch, fbgemm_gpu, torchrec, polars, xxhash, gin; print('imports OK,', torch.__version__, torch.cuda.device_count(),'gpus')" 2>&1 | tee -a "$LOG"

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WORLD_SIZE=$(python -c "import torch; print(torch.cuda.device_count())")
# AMD/ROCm: Triton HSTU kernel hits PassManager errors on some shapes; force
# PYTORCH backend. On CUDA, unset this to default to TRITON for ~3-5x speedup.
export HSTU_HAMMER_KERNEL=${HSTU_HAMMER_KERNEL:-PYTORCH}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# --- GPU clock sanity guard ---------------------------------------------------
# Leftover node state once pinned all 8 GPUs into `perf_determinism` at half
# clock (1093 vs 2200 MHz max). That uniformly slowed every Triton kernel ~1.9x
# and silently masked real perf changes for an entire debugging session. Always
# log the perf level + a live sclk sample so a capped run is obvious from the
# log, and try to restore boost. Fully non-fatal (rocm-smi may be absent or
# lack permission inside the container — in that case reset from the host).
if command -v rocm-smi >/dev/null 2>&1; then
  echo "[$(date)] GPU perf-level check:" | tee -a "$LOG"
  rocm-smi --showperflevel 2>/dev/null | grep -iE "GPU\[[0-9]+\]" | tee -a "$LOG" || true
  if rocm-smi --showperflevel 2>/dev/null | grep -iqE "Performance Level: *(perf_determinism|manual|low)"; then
    echo "[$(date)] WARNING: GPUs not in 'auto' perf level — attempting --setperflevel auto" | tee -a "$LOG"
    rocm-smi --setperflevel auto 2>/dev/null | grep -iE "set to auto" | tee -a "$LOG" \
      || echo "[$(date)] WARNING: could not set perf level (no permission?). Run 'rocm-smi --setperflevel auto' on the HOST before benchmarking — clocks may be capped." | tee -a "$LOG"
  fi
  echo "[$(date)] sclk sample (GPU0):$(rocm-smi -d 0 --showclocks 2>/dev/null | grep -i 'sclk clock level' | sed -E 's/.*sclk clock level//')" | tee -a "$LOG" || true
fi
# -----------------------------------------------------------------------------

echo "[$(date)] launching train_ranker with WORLD_SIZE=$WORLD_SIZE" | tee -a "$LOG"

python -m generative_recommenders.dlrm_v3.train.train_ranker \
    --dataset yambda-5b --mode train-eval 2>&1 | tee -a "$LOG"
