#!/bin/bash
#SBATCH --job-name=yambda_slurm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=meta64          # [CLUSTER-SPECIFIC] partition name
#SBATCH --time=01:10:00
#SBATCH --output=yambda_slurm.%j.out
# ^ relative to the submit dir (SLURM parses #SBATCH before any shell runs, so it
#   cannot expand env vars). The real consolidated run log is $LOG (see below),
#   which defaults under $SCRATCH; this file just captures the batch stdout.
# =============================================================================
# launch_slurm.sh — single entry point for the yambda-5b trainer on N>=1 nodes.
#
# Consolidates what used to be separate scripts so multi-node enablement is
# ONE committable script (plus the train_ranker.py / utils.py python changes):
#   * orchestrate phase (host SLURM glue) — formerly sbatch_smoke_multinode.sh
#   * provision   phase (container + RDMA) — formerly _provision_yambda_primus.sh
#   * worker      phase (in-container train) — now inlined below
#
# PHASES (auto-detected from context; force with LAUNCH_SLURM_PHASE=<phase>):
#   orchestrate  Runs on the SLURM batch host (no /.dockerenv). Resolves the
#                rendezvous (MASTER_ADDR/PORT), ensures the container on every
#                node (provision phase), then `docker exec`s the worker phase on
#                every node, one task per node.
#   provision    Runs on a compute-node host. Ensures the `yambda_primus`
#                container is up (loads the pre-baked image if present — no
#                internet/pip — else builds from the base image) and stages the
#                host RDMA userspace overlay on shared NFS.
#   worker       Runs INSIDE the container. Sets the distributed topology +
#                NCCL/RDMA env and spawns this node's GPU ranks via train_ranker.
#                N==1 transparently uses the legacy single-node path (localhost,
#                node_rank 0), byte-for-byte as before, so the streaming-e2e
#                supervisor's direct `bash scripts/launch_slurm.sh` is unchanged.
#
# USAGE
#   Reference run (1 node): sbatch --nodes=1 scripts/launch_slurm.sh
#   Reference run (N node):  sbatch --nodes=N scripts/launch_slurm.sh
#     ^ a bare submit reproduces the FROZEN REFERENCE shape (full 299-window
#       sweep + data-fraction eval cadence). Prepend SMOKE=1 for a fast
#       functional check (short window, capped batches).
#   Single-node direct: bash scripts/launch_slurm.sh   (already inside container;
#                       what run_streaming_e2e.sh invokes per relaunch — uses the
#                       gin defaults, NOT the orchestrate reference shape)
#   Perf pair:
#     LOG=/apps/chcai/perf_1node.log NUM_TRAIN_BATCHES=200 NUM_EVAL_BATCHES=0 \
#       EVAL_EACH_WINDOW=0 METRIC_LOG_FREQ=20 \
#       sbatch --nodes=1 --job-name=y1 scripts/launch_slurm.sh
#     LOG=/apps/chcai/perf_2node.log NUM_TRAIN_BATCHES=200 NUM_EVAL_BATCHES=0 \
#       EVAL_EACH_WINDOW=0 METRIC_LOG_FREQ=20 \
#       sbatch --nodes=2 --job-name=y2 scripts/launch_slurm.sh
#     # then: bash scripts/compare_node_perf.sh /apps/chcai/perf_1node.log /apps/chcai/perf_2node.log
#
# ONE-TIME IMAGE BAKE (so fresh nodes skip the multi-GB torch download + pip):
#   BAKE_IMAGE=1 LAUNCH_SLURM_PHASE=provision bash scripts/launch_slurm.sh
#   (commits the deps-installed container to $BAKED_IMAGE and `docker save`s it to
#    $BAKED_TAR on NFS; subsequent provisions `docker load` it offline.)
#
# -----------------------------------------------------------------------------
# PORTABILITY — what to change for a DIFFERENT cluster / network / hardware.
# Every such knob is also tagged inline with "[CLUSTER-SPECIFIC]" (grep for it).
# All are env-overridable, so you can adapt without editing this file.
#
#  A) SLURM / scheduler
#     - #SBATCH --partition=meta64  : partition name. CHANGE per cluster.
#     - #SBATCH --time / --exclusive : policy; adjust to taste.
#
#  B) Filesystems (must be shared/NFS across ALL nodes — this script re-invokes
#     itself and reads the overlay + data from these paths cluster-wide)
#     - REPO_MOUNT (repo + this script, e.g. /home/<user>) is bind-mounted rw;
#       DATA_MOUNT (e.g. /apps/chcai) holds the read-only dataset + overlay +
#       baked tar + pip tarball; SCRATCH (e.g. /home/<user>/yambda_runs) is the
#       writable log/output root. Override any via env — nothing is user-hardwired.
#
#  C) Container image / GPU software stack (tied to the GPU arch + ROCm version)
#     - IMAGE=rocm/primus:v26.3        : base image. ROCm/AMD-specific.
#     - docker run --device=/dev/kfd --device=/dev/dri --group-add video : AMD ROCm
#       device passthrough. For NVIDIA this is --gpus all / nvidia runtime instead.
#     - --ulimit memlock=-1 : REQUIRED for RDMA QP registration (do not drop).
#     - TORCH_IDX (rocm7.2), torch/vision/audio ==*+rocm7.2, FBGEMM_WHL (a gfx950
#       wheel), torchrec pin : the whole deps set is arch/ROCm-version-specific.
#
#  D) Network fabric — THE trickiest part; defaults are PROVEN on meta64 cv350
#     (Broadcom bnxt_re RoCEv2). On a different fabric these almost certainly change
#     (see the worker-phase block for the full rationale):
#     - NCCL_SOCKET_IFNAME=fenic0 : the ONE routable host NIC for TCP bootstrap.
#       Find yours with `ip -br addr`; the per-GPU RDMA NICs are usually NOT
#       routable for plain TCP, so auto-detect hangs init — you MUST pin this.
#     - NCCL_IB_HCA=bnxt_re0..7 : the RDMA HCA device names. List with `ibv_devices`.
#       Different NIC vendor (e.g. mlx5_*, ionic_*) => different names AND a
#       different userspace provider, which changes the RDMA overlay below.
#     - NCCL_IB_GID_INDEX=3 : RoCEv2 IPv4 GID index. Check `show_gids`; v1/v2 and
#       IPv4/IPv6 live at different indices per port.
#     - NCCL_IB_TC=104 : RoCE lossless (PFC) traffic class. Fabric/switch-specific.
#     - RDMA overlay (provision phase): only needed when the CONTAINER's rdma-core
#       is older than the HOST kernel driver's uapi (our bnxt_re v34-vs-v59 case).
#       Different NIC/host => different /usr/lib64 provider .so to stage, or the
#       overlay may be unnecessary entirely (set RDMA_OVERLAY= to disable). If RDMA
#       can't be made to work, NCCL_NET_TRANSPORT=socket falls back to TCP.
#
#  E) Not cluster-specific (auto-derived): GPUS_PER_NODE (torch.cuda.device_count),
#     NNODES/NODE_RANK/MASTER_ADDR (from SLURM), WORLD_SIZE.
# =============================================================================
set -uo pipefail

# Absolute path to THIS script so the orchestrate phase can re-invoke it on every
# node (home is shared NFS, so the same path resolves cluster-wide).
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

# ---- phase detection --------------------------------------------------------
PHASE="${LAUNCH_SLURM_PHASE:-}"
if [ -z "$PHASE" ]; then
  if [ -f /.dockerenv ]; then PHASE=worker; else PHASE=orchestrate; fi
fi

# ---- shared config (env-overridable) ----------------------------------------
CONTAINER=${CONTAINER:-yambda_${USER:-$(id -un)}}   # per-user container name (do NOT reuse another user's container — its bind mounts differ)
REPO=${REPO:-$REPO_ROOT}                       # repo path inside the container
IMAGE=${IMAGE:-rocm/primus:v26.3}              # [CLUSTER-SPECIFIC] ROCm/arch base image
BAKED_IMAGE=${BAKED_IMAGE:-yambda_primus_baked:latest}
BAKED_TAR=${BAKED_TAR:-/apps/chcai/yambda_primus_baked.tar}   # [CLUSTER-SPECIFIC] shared-NFS path (read-only build asset)
USE_BAKED=${USE_BAKED:-1}
OVERLAY=${RDMA_OVERLAY:-/apps/chcai/rdma_host_el9_new}        # [CLUSTER-SPECIFIC] shared-NFS RDMA overlay (read-only, already staged)

# Bind mounts + scratch — all on shared NFS, identical path on every node.
#   REPO_MOUNT : NFS home root that contains THIS repo (bind-mounted rw).
#   DATA_MOUNT : NFS root with the (shared, read-only) dataset + RDMA overlay +
#                pip/fbgemm build assets. Kept as-is so the dataset is NOT
#                duplicated. You only need read access here.
#   SCRATCH    : this run's WRITABLE output root (logs / tb / traces).
# All env-overridable, so nothing is hardwired to one user's home.
REPO_MOUNT=${REPO_MOUNT:-$HOME}              # NFS home holding the repo (must contain $REPO); override if your repo lives elsewhere
DATA_MOUNT=${DATA_MOUNT:-/apps/chcai}        # shared dataset + RDMA overlay + pip/fbgemm assets (read-only)
SCRATCH=${SCRATCH:-$HOME/yambda_runs}        # writable output root (logs / tb / traces)

# =============================================================================
# PHASE: orchestrate  (SLURM batch host)
# =============================================================================
orchestrate() {
  # When run as the SLURM batch script, $0 is the node-local staged copy
  # (/var/spool/slurmd/job<ID>/slurm_script), so $SELF / $REPO_ROOT are WRONG
  # here (they don't exist on other nodes). Resolve the REAL shared-NFS script
  # path + repo root from SLURM so we can re-invoke this script on every node and
  # `cd` to the right repo inside the container.
  SCRIPT_PATH=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -oP 'Command=\K\S+')
  [ -f "${SCRIPT_PATH:-}" ] || SCRIPT_PATH="${SLURM_SUBMIT_DIR:-$REPO_ROOT}/scripts/launch_slurm.sh"
  [ -f "$SCRIPT_PATH" ] || SCRIPT_PATH="$SELF"
  REPO=$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)

  mkdir -p "$SCRATCH" 2>/dev/null || true
  LOG=${LOG:-$SCRATCH/yambda_slurm.${SLURM_JOB_ID:-manual}.log}

  # Run-shape defaults. By DEFAULT a bare `sbatch scripts/launch_slurm.sh`
  # reproduces the FROZEN REFERENCE run: full 299-window sweep (START_TS=0) with
  # the data-fraction eval cadence (eval every 0.5% of the training stream). Set
  # SMOKE=1 for a fast functional check (short dense window, capped batches,
  # per-window eval). Any individual knob below stays env-overridable.
  MODE=${MODE:-streaming-train-eval}
  if [ "${SMOKE:-0}" = "1" ]; then
    START_TS=${START_TS:-150}
    NUM_TRAIN_TS=${NUM_TRAIN_TS:-1}
    NUM_TRAIN_BATCHES=${NUM_TRAIN_BATCHES:-20}
    NUM_EVAL_BATCHES=${NUM_EVAL_BATCHES:-10}
    EVAL_EVERY_N_WINDOWS=${EVAL_EVERY_N_WINDOWS:-1}
    METRIC_LOG_FREQ=${METRIC_LOG_FREQ:-5}
  fi
  START_TS=${START_TS:-0}
  NUM_TRAIN_TS=${NUM_TRAIN_TS:-299}
  NUM_TRAIN_BATCHES=${NUM_TRAIN_BATCHES:-0}
  NUM_EVAL_BATCHES=${NUM_EVAL_BATCHES:-0}
  EVAL_EACH_WINDOW=${EVAL_EACH_WINDOW:-1}
  METRIC_LOG_FREQ=${METRIC_LOG_FREQ:-20}
  # Eval cadence — the two knobs are mutually exclusive (the worker raises if both
  # are >0). Data-fraction is the reference default; if the caller explicitly
  # selected the per-window cadence (EVAL_EVERY_N_WINDOWS>0) leave data-pct off,
  # otherwise default to the reference 0.5%-of-data cadence (per-window disabled).
  if [ "${EVAL_EVERY_N_WINDOWS:-0}" -gt 0 ] 2>/dev/null; then
    EVAL_EVERY_DATA_PCT=${EVAL_EVERY_DATA_PCT:-0}
  else
    EVAL_EVERY_N_WINDOWS=0
    EVAL_EVERY_DATA_PCT=${EVAL_EVERY_DATA_PCT:-0.005}
  fi
  FORCE_PROVISION=${FORCE_PROVISION:-0}

  # Truncate the metrics log on a FRESH run; APPEND on a supervised relaunch
  # (APPEND_LOG=1) so the full-run NE/AUC history survives crash/node-failover
  # resubmits instead of being wiped on every attempt (mirrors the single-node
  # supervisor's init-once/append model).
  if [ "${APPEND_LOG:-0}" = "1" ]; then
    echo "[$(date)] === resume: appending to existing $LOG (APPEND_LOG=1) ===" >> "$LOG"
  else
    : > "$LOG"
  fi
  # World-writable so the in-container worker (running as root, squashed to
  # `nobody` over root-squashed NFS) can append via `tee -a $LOG`. Without this
  # the worker's tee opens the file read-only-denied and exits non-zero, which
  # pipefail turns into a spurious rc=1 even when training succeeds.
  chmod 666 "$LOG" 2>/dev/null || true
  echo "[$(date)] launch_slurm/orchestrate: job=${SLURM_JOB_ID:-?} nodes=${SLURM_JOB_NODELIST:-?} nnodes=${SLURM_NNODES:-1}" | tee -a "$LOG"
  echo "[$(date)] resolved SCRIPT_PATH=$SCRIPT_PATH REPO=$REPO" | tee -a "$LOG"
  echo "[$(date)] config: MODE=$MODE START_TS=$START_TS NUM_TRAIN_TS=$NUM_TRAIN_TS NUM_TRAIN_BATCHES=$NUM_TRAIN_BATCHES NUM_EVAL_BATCHES=$NUM_EVAL_BATCHES METRIC_LOG_FREQ=$METRIC_LOG_FREQ SMOKE=${SMOKE:-0} EVAL_EVERY_N_WINDOWS=$EVAL_EVERY_N_WINDOWS EVAL_EVERY_DATA_PCT=$EVAL_EVERY_DATA_PCT" | tee -a "$LOG"
  echo "[$(date)] lr-override: DENSE_LR=${DENSE_LR:-<unset:gin default 1e-7>} SPARSE_LR=${SPARSE_LR:-<unset:gin default 1e-7>}" | tee -a "$LOG"

  # Rendezvous resolved on the HOST (the container image has no SLURM client).
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1)
  MASTER_ADDR=${MASTER_ADDR:-localhost}
  MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))
  echo "[$(date)] rendezvous: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT" | tee -a "$LOG"

  # Optional NCCL/RCCL fabric overrides — forwarded into the container only when
  # set at submit time (docker exec does NOT inherit the srun task env). The
  # worker phase applies its own validated multi-node bnxt_re defaults when these
  # are unset. Common: NCCL_NET_TRANSPORT=socket (TCP fallback), NCCL_DEBUG=INFO.
  NCCL_ENV_ARGS=""
  for v in NCCL_NET_TRANSPORT NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_IB_GID_INDEX \
           NCCL_IB_TC NCCL_IB_TIMEOUT NCCL_IGNORE_CPU_AFFINITY RCCL_MSCCL_ENABLE NCCL_NET_GDR_LEVEL \
           NCCL_IB_PCI_RELAXED_ORDERING NCCL_IB_USE_INLINE NCCL_IB_QPS_PER_CONNECTION \
           NCCL_IB_ECE_ENABLE NCCL_DMABUF_ENABLE NCCL_GDRCOPY_ENABLE NCCL_GDR_FLUSH_DISABLE \
           NCCL_PXN_DISABLE NCCL_CHECKS_DISABLE NCCL_CROSS_NIC RDMA_OVERLAY; do
    eval "val=\${$v:-}"
    if [ -n "$val" ]; then NCCL_ENV_ARGS="$NCCL_ENV_ARGS -e $v=$val"; fi
  done

  # TRICKY — variable expansion inside the `srun ... bash -c "..."` blocks below:
  # the string is double-quoted, so PLAIN $VAR expands NOW on the batch host (e.g.
  # $MASTER_ADDR, $CONTAINER, $SCRIPT_PATH — values computed above), while
  # BACKSLASH-escaped \$VAR is passed through literally and expands LATER on each
  # compute node inside the srun task (e.g. \$SLURM_NODEID, \$(hostname)) where the
  # per-node SLURM_* env actually lives. Mixing these up sends every rank the
  # wrong node id or breaks the docker exec — keep the \$ on per-node values.

  # --- step 1: ensure the container is up on every node ----------------------
  echo "[$(date)] ensuring container '$CONTAINER' on all nodes (force=$FORCE_PROVISION)" | tee -a "$LOG"
  srun --ntasks-per-node=1 bash -c "
    # Reap stale/foreign GPU containers from prior jobs BEFORE (re)provisioning.
    # The node is allocated --exclusive, so any GPU container other than
    # '$CONTAINER' is an orphan left by a previous job (its container outlives the
    # SLURM allocation). We remove every such container that has GPU access
    # (/dev/kfd or /dev/dri) — running OR stopped, whether or not it currently
    # pins VRAM ('docker ps -aq' includes stopped ones) — since idle orphans can
    # still hold device handles or wake up; leaked HBM from these has caused both
    # OOMs and RCCL collective hangs. We deliberately SKIP non-GPU containers
    # (e.g. 'k8s-node-services-*' and other cluster system services) so we don't
    # disrupt node infrastructure. docker teardown lets the driver reclaim HBM.
    for _c in \$(docker ps -aq 2>/dev/null); do
      _nm=\$(docker inspect -f '{{.Name}}' \"\$_c\" 2>/dev/null | sed 's#^/##')
      [ \"\$_nm\" = \"$CONTAINER\" ] && continue
      _dev=\$(docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' \"\$_c\" 2>/dev/null)
      case \"\$_dev\" in
        *kfd*|*dri*)
          echo \"[\$(hostname)] reaping stale GPU container \$_nm (\$_c)\"
          docker rm -f \"\$_c\" >/dev/null 2>&1 || true ;;
        *)
          echo \"[\$(hostname)] keeping non-GPU/system container \$_nm (\$_c)\" ;;
      esac
    done
    # Reuse a STOPPED '$CONTAINER' (its installed deps persist in the container
    # fs) instead of destructively re-provisioning from the base image + pip.
    # Harmless no-op on a fresh node (no such container) -> falls through to
    # provision below. Repo code is bind-mounted, so live edits are still picked up.
    docker start $CONTAINER >/dev/null 2>&1 || true
    if [ \"$FORCE_PROVISION\" = \"1\" ] || ! docker exec $CONTAINER true >/dev/null 2>&1; then
      echo \"[\$(hostname)] (re)provisioning container\"
      LAUNCH_SLURM_PHASE=provision CONTAINER=$CONTAINER IMAGE=$IMAGE \
        BAKED_IMAGE=$BAKED_IMAGE BAKED_TAR=$BAKED_TAR USE_BAKED=$USE_BAKED \
        BAKE_IMAGE=${BAKE_IMAGE:-0} RDMA_OVERLAY=$OVERLAY REPO=$REPO \
        REPO_MOUNT=$REPO_MOUNT DATA_MOUNT=$DATA_MOUNT SCRATCH=$SCRATCH bash $SCRIPT_PATH
    else
      # Container persists across jobs; the reap above only removes FOREIGN GPU
      # containers, so our own '$CONTAINER' can still pin HBM via stray trainer
      # ranks left by a prior OOM/crash (this caused repeated 'CUDA out of memory'
      # on relaunch onto the same node). Restart it to kill every exec'd proc and
      # let the driver reclaim HBM — cheap (keeps the installed deps in the
      # container fs; NFS RDMA overlay also persists), no full re-provision.
      echo \"[\$(hostname)] container already up — restarting to free any leaked HBM before launch\"
      docker restart $CONTAINER >/dev/null 2>&1 || true
      # Readiness gate: a bare 'docker exec true' can pass while the runtime is
      # still settling, so the SUBSEQUENT (heavier) worker exec races the restart
      # and dies with 'container is not running' / OCI 'setns' errors (observed on
      # c07-08 and e08-08 -> the peer never joins rendezvous -> master 600s
      # TCPStore timeout). Require State.Running=true AND a successful probe, then
      # a short settle, before considering the container ready.
      for _w in \$(seq 1 30); do
        [ \"\$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null)\" = \"true\" ] \
          && docker exec $CONTAINER true >/dev/null 2>&1 && break
        sleep 2
      done
      sleep 2
      echo \"[\$(hostname)] container restarted (HBM reclaimed; running=\$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null))\"
    fi
  " 2>&1 | tee -a "$LOG"

  # --- step 2: launch the worker (trainer) inside the container on every node -
  echo "[$(date)] launching trainer (worker phase) on all nodes" | tee -a "$LOG"
  srun --ntasks-per-node=1 bash -c "
    # Pre-flight readiness gate (per node): step 1 ran in a SEPARATE srun, so the
    # container can still be settling here. Wait for State.Running=true + a probe
    # before the worker exec so we don't race a just-restarted container.
    for _w in \$(seq 1 30); do
      [ \"\$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null)\" = \"true\" ] \
        && docker exec $CONTAINER true >/dev/null 2>&1 && break
      [ \$_w -eq 1 ] && echo \"[\$(hostname)] worker pre-flight: waiting for container to be ready...\"
      sleep 2
    done
    # Retry wrapper: docker exec startup failures (rc 125 daemon 'container is not
    # running', 126/127 OCI/setns 'exec failed') mean the container wasn't ready,
    # NOT that the trainer ran and failed. Restart + re-gate + retry a few times.
    # Any OTHER rc (the trainer actually started and exited) is propagated so the
    # supervisor's resume-from-checkpoint logic owns real failures.
    _wattempt=0
    while : ; do
    _wattempt=\$((_wattempt+1))
    docker exec \
      -e LAUNCH_SLURM_PHASE=worker \
      -e WORKER_TEE=0 \
      -e SCRATCH=$SCRATCH \
      -e SLURM_NNODES=\$SLURM_NNODES \
      -e SLURM_NODEID=\$SLURM_NODEID \
      -e SLURM_PROCID=\$SLURM_PROCID \
      -e SLURM_JOB_NODELIST=\"\$SLURM_JOB_NODELIST\" \
      -e SLURM_JOB_ID=\$SLURM_JOB_ID \
      -e MASTER_ADDR=$MASTER_ADDR \
      -e MASTER_PORT=$MASTER_PORT \
      -e HSTU_HAMMER_KERNEL=${HSTU_HAMMER_KERNEL:-TRITON} \
      -e MODE=$MODE \
      -e START_TS=$START_TS \
      -e NUM_TRAIN_TS=$NUM_TRAIN_TS \
      -e EVAL_EACH_WINDOW=$EVAL_EACH_WINDOW \
      -e EVAL_EVERY_N_WINDOWS=$EVAL_EVERY_N_WINDOWS \
      ${EVAL_EVERY_DATA_PCT:+-e EVAL_EVERY_DATA_PCT=$EVAL_EVERY_DATA_PCT} \
      -e NUM_TRAIN_BATCHES=$NUM_TRAIN_BATCHES \
      -e NUM_EVAL_BATCHES=$NUM_EVAL_BATCHES \
      -e METRIC_LOG_FREQ=$METRIC_LOG_FREQ \
      ${STREAMING_SHUFFLE_FRACTION:+-e STREAMING_SHUFFLE_FRACTION=$STREAMING_SHUFFLE_FRACTION} \
      ${STREAMING_SHUFFLE_SEED:+-e STREAMING_SHUFFLE_SEED=$STREAMING_SHUFFLE_SEED} \
      ${NUM_WORKERS:+-e NUM_WORKERS=$NUM_WORKERS} \
      ${PREFETCH_FACTOR:+-e PREFETCH_FACTOR=$PREFETCH_FACTOR} \
      ${DIAG_UNIQUE_EMB:+-e DIAG_UNIQUE_EMB=$DIAG_UNIQUE_EMB} \
      ${DIAG_EMB_STEPS:+-e DIAG_EMB_STEPS=$DIAG_EMB_STEPS} \
      ${OUTPUT_TRACE:+-e OUTPUT_TRACE=$OUTPUT_TRACE} \
      ${MIN_HISTORY:+-e MIN_HISTORY=$MIN_HISTORY} \
      ${HISTORY_STRATEGY:+-e HISTORY_STRATEGY=$HISTORY_STRATEGY} \
      ${SEED:+-e SEED=$SEED} \
      ${DENSE_LR:+-e DENSE_LR=$DENSE_LR} \
      ${SPARSE_LR:+-e SPARSE_LR=$SPARSE_LR} \
      ${GRAD_CLIP_NORM:+-e GRAD_CLIP_NORM=$GRAD_CLIP_NORM} \
      ${HSTU_NUM_LAYERS:+-e HSTU_NUM_LAYERS=$HSTU_NUM_LAYERS} \
      ${MAX_SEQ_LEN:+-e MAX_SEQ_LEN=$MAX_SEQ_LEN} \
      ${HISTORY_LENGTH:+-e HISTORY_LENGTH=$HISTORY_LENGTH} \
      ${BATCH_SIZE:+-e BATCH_SIZE=$BATCH_SIZE} \
      ${CKPT_TIME_INTERVAL_S:+-e CKPT_TIME_INTERVAL_S=$CKPT_TIME_INTERVAL_S} \
      ${KEEP_LAST_N:+-e KEEP_LAST_N=$KEEP_LAST_N} \
      ${IN_WINDOW_CKPT_FREQ:+-e IN_WINDOW_CKPT_FREQ=$IN_WINDOW_CKPT_FREQ} \
      ${CKPT_STEP_FREQ:+-e CKPT_STEP_FREQ=$CKPT_STEP_FREQ} \
      -e TRAIN_SPLIT_PERCENTAGE=${TRAIN_SPLIT_PERCENTAGE:-1.0} \
      -e AUC_THRESHOLD=${AUC_THRESHOLD:-0.80275} \
      ${EVAL_ACCURACY_AUC_MODE:+-e EVAL_ACCURACY_AUC_MODE=$EVAL_ACCURACY_AUC_MODE} \
      -e SPLIT_SALT=${SPLIT_SALT:-0} \
      -e EVAL_HOLDOUT_TS=${EVAL_HOLDOUT_TS:--1} \
      -e EVAL_HOLDOUT_NUM_WINDOWS=${EVAL_HOLDOUT_NUM_WINDOWS:-1} \
      ${WORKER_CMD:+-e WORKER_CMD=\"$WORKER_CMD\"} \
      ${RUN_NAME:+-e RUN_NAME=$RUN_NAME} \
      ${TENSORBOARD_LOG_PATH:+-e TENSORBOARD_LOG_PATH=$TENSORBOARD_LOG_PATH} \
      ${MLPERF_LOG_PATH:+-e MLPERF_LOG_PATH=$MLPERF_LOG_PATH} \
      ${CKPT_PATH:+-e CKPT_PATH=$CKPT_PATH} \
      ${SPARSE_A2A_FWD:+-e SPARSE_A2A_FWD=$SPARSE_A2A_FWD} \
      ${SPARSE_A2A_BWD:+-e SPARSE_A2A_BWD=$SPARSE_A2A_BWD} \
      -e LOG=$LOG \
      $NCCL_ENV_ARGS \
      $CONTAINER bash -lc 'cd $REPO && LAUNCH_SLURM_PHASE=worker bash scripts/launch_slurm.sh'
    _wrc=\$?
    if { [ \$_wrc -eq 125 ] || [ \$_wrc -eq 126 ] || [ \$_wrc -eq 127 ]; } && [ \$_wattempt -lt 5 ]; then
      echo \"[\$(hostname)] worker exec failed to START (rc=\$_wrc, attempt \$_wattempt/5) — container not ready; restarting + retrying\"
      docker restart $CONTAINER >/dev/null 2>&1 || true
      for _w in \$(seq 1 30); do
        [ \"\$(docker inspect -f '{{.State.Running}}' $CONTAINER 2>/dev/null)\" = \"true\" ] \
          && docker exec $CONTAINER true >/dev/null 2>&1 && break
        sleep 2
      done
      sleep 3
      continue
    fi
    exit \$_wrc
    done
  " 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "[$(date)] launch_slurm/orchestrate finished rc=$rc" | tee -a "$LOG"
  exit $rc
}

# =============================================================================
# PHASE: provision  (compute-node host)
# =============================================================================
provision() {
  export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
  DOCKER=$(command -v docker 2>/dev/null || true); DOCKER=${DOCKER:-/usr/bin/docker}
  FBGEMM_WHL=${FBGEMM_WHL:-/apps/chcai/FBGEMM/fbgemm_gpu/dist/fbgemm_gpu_nightly_rocm-2026.6.2-cp312-cp312-linux_x86_64.whl}  # [CLUSTER-SPECIFIC] gfx950/ROCm wheel
  TORCH_IDX=${TORCH_IDX:-https://download.pytorch.org/whl/rocm7.2}  # [CLUSTER-SPECIFIC] ROCm version index
  echo "[provision] host=$(hostname) container=$CONTAINER docker=$DOCKER"

  # Resolve which image to run + whether deps must be installed. Prefer a pre-baked
  # image (deps already installed) to skip the multi-GB torch download + pip /
  # torchrec-from-git build on every fresh node:
  #   1) baked image in this node's docker -> use it, skip deps
  #   2) baked image tar on NFS            -> docker load (local, no internet)
  #   3) neither                           -> base image + pip (slow path, which
  #                                           can then be baked via BAKE_IMAGE=1)
  NEED_DEPS=1
  RUN_IMAGE="$IMAGE"
  if [ "$USE_BAKED" = "1" ]; then
    if "$DOCKER" image inspect "$BAKED_IMAGE" >/dev/null 2>&1; then
      echo "[provision] using baked image $BAKED_IMAGE (deps preinstalled, no download)"
      RUN_IMAGE="$BAKED_IMAGE"; NEED_DEPS=0
    elif [ -f "$BAKED_TAR" ]; then
      echo "[provision] loading baked image from $BAKED_TAR (local, no internet)..."
      if "$DOCKER" load -i "$BAKED_TAR" >/dev/null 2>&1 && "$DOCKER" image inspect "$BAKED_IMAGE" >/dev/null 2>&1; then
        RUN_IMAGE="$BAKED_IMAGE"; NEED_DEPS=0; echo "[provision] baked image loaded"
      else
        echo "[provision] WARNING: docker load failed; falling back to base-image + pip"
      fi
    fi
  fi
  if ! "$DOCKER" image inspect "$RUN_IMAGE" >/dev/null 2>&1; then
    echo "[provision] pulling $RUN_IMAGE (this can take a while)..."; "$DOCKER" pull "$RUN_IMAGE"
  fi

  echo "[provision] (re)starting container $CONTAINER from $RUN_IMAGE"
  "$DOCKER" rm -f "$CONTAINER" >/dev/null 2>&1 || true
  "$DOCKER" run -d --name "$CONTAINER" \
    --network=host --ipc=host --shm-size=64g \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    `# [CLUSTER-SPECIFIC] AMD ROCm device passthrough; NVIDIA uses --gpus all / nvidia runtime` \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
    `# memlock=-1 is REQUIRED for RDMA QP memory registration — do not drop` \
    --security-opt seccomp=unconfined --privileged \
    -v "$REPO_MOUNT:$REPO_MOUNT" \
    -v "$DATA_MOUNT:$DATA_MOUNT" \
    `# shared-NFS bind mounts: repo home (REPO_MOUNT, rw) + dataset/build assets (DATA_MOUNT)` \
    -w "$REPO" \
    "$RUN_IMAGE" sleep infinity

  # --- RDMA userspace overlay for in-container RCCL (bnxt_re) -----------------
  # The image (rocm/primus, rdma-core 50/libbnxt_re-rdmav34) ships an OLDER RDMA
  # userspace than the host kernel bnxt_re driver. The stock v34 provider faults
  # RCCL's deep-queue create_qp (max_send_wr=256) against the newer kernel uapi
  # -> "ibv_create_qp ... Bad address". Fix: stage the host's matched rdma-core
  # (libibverbs v61 + libbnxt_re-rdmav59 + libnl) on NFS so the worker phase makes
  # RCCL load it via LD_PRELOAD + LD_LIBRARY_PATH. The UNVERSIONED libibverbs.so
  # symlink is essential (import torch pulls the unversioned soname; without it
  # the lookup falls through to the container v34 lib and the fix regresses).
  if [ "${FORCE_OVERLAY:-0}" != "1" ] && ls "$OVERLAY/lib/libibverbs/"libbnxt_re-rdmav*.so >/dev/null 2>&1 && [ -L "$OVERLAY/lib/libibverbs.so" ]; then
    echo "[provision] host RDMA overlay already staged at $OVERLAY (shared NFS) — skipping"
  else
    echo "[provision] staging host RDMA userspace overlay -> $OVERLAY"
    rm -rf "${OVERLAY}.tmp" 2>/dev/null
    mkdir -p "${OVERLAY}.tmp/lib/libibverbs" "${OVERLAY}.tmp/libibverbs.d"
    cp -L /usr/lib64/libibverbs.so.1 /usr/lib64/libnl-3.so.200 /usr/lib64/libnl-route-3.so.200 "${OVERLAY}.tmp/lib/" 2>/dev/null || true
    ln -sf libibverbs.so.1 "${OVERLAY}.tmp/lib/libibverbs.so"
    cp -L /usr/lib64/libibverbs/*.so "${OVERLAY}.tmp/lib/libibverbs/" 2>/dev/null || true
    cp /etc/libibverbs.d/*.driver "${OVERLAY}.tmp/libibverbs.d/" 2>/dev/null || true
    if ls "${OVERLAY}.tmp/lib/libibverbs/"libbnxt_re-rdmav*.so >/dev/null 2>&1; then
      rm -rf "$OVERLAY" 2>/dev/null
      mv "${OVERLAY}.tmp" "$OVERLAY" 2>/dev/null || { mkdir -p "$OVERLAY"; cp -a "${OVERLAY}.tmp/." "$OVERLAY/"; }
      echo "[provision] host RDMA overlay staged: $(ls "$OVERLAY/lib/libibverbs" | wc -l) providers + libibverbs.so symlink"
    else
      echo "[provision] WARNING: host bnxt_re provider not found at /usr/lib64/libibverbs — multi-node RDMA will fail 'Bad address'; use NCCL_NET_TRANSPORT=socket"
    fi
  fi

  if [ "$NEED_DEPS" = "0" ]; then
    echo "[provision] baked image — deps preinstalled; verifying imports only"
    "$DOCKER" exec "$CONTAINER" bash -lc '
python -c "import torch, fbgemm_gpu, torchrec, polars, xxhash, gin; print(\"imports OK,\", torch.__version__, torch.version.hip, torch.cuda.device_count(), \"gpus\")"
' || echo "[provision] WARNING: baked-image import smoke failed"
  else
    echo "[provision] installing recipe deps (base image, slow path)"
    # Install misc deps FIRST, then pin the rocm torch stack + fbgemm + torchrec
    # LAST with --no-deps so nothing pulls a CUDA torch over the rocm build.
    "$DOCKER" exec "$CONTAINER" bash -lc '
set -e
echo "=== native torch ==="; python -c "import torch;print(torch.__version__)" || true
echo "=== misc python deps ==="
pip install --no-cache-dir polars-u64-idx pyarrow pyyaml tqdm psutil numba xxhash gin-config \
  absl-py pandas tensorboard torchmetrics tensordict pyre-extensions iopath typing-inspect 2>&1 | tail -3 || true
echo "=== rocm torch stack (force, no-deps, LAST) ==="
pip install --force-reinstall --no-deps --index-url '"$TORCH_IDX"' \
  torch==2.12.0+rocm7.2 torchvision==0.27.0+rocm7.2 torchaudio==2.11.0+rocm7.2
echo "=== fbgemm (local gfx950 wheel) ==="
pip install --force-reinstall --no-deps '"$FBGEMM_WHL"'
echo "=== torchrec v2026.06.01.00 (force, no-deps) ==="
pip install --force-reinstall --no-deps "git+https://github.com/pytorch/torchrec.git@v2026.06.01.00"
echo "=== import smoke ==="
python -c "import torch, fbgemm_gpu, torchrec, polars, xxhash, gin; print(\"imports OK,\", torch.__version__, torch.version.hip, torch.cuda.device_count(), \"gpus\")"
'
  fi

  # --- one-time bake: snapshot the deps-installed container into a reusable image
  # and save it to NFS so future nodes skip the download/pip path entirely.
  if [ "${BAKE_IMAGE:-0}" = "1" ]; then
    echo "[provision] baking: docker commit $CONTAINER -> $BAKED_IMAGE"
    if "$DOCKER" commit "$CONTAINER" "$BAKED_IMAGE" >/dev/null; then
      echo "[provision] saving $BAKED_IMAGE -> $BAKED_TAR (one-time, tens of GB)"
      if "$DOCKER" save "$BAKED_IMAGE" -o "${BAKED_TAR}.tmp.$$" && mv -f "${BAKED_TAR}.tmp.$$" "$BAKED_TAR"; then
        echo "[provision] bake done: $(ls -lh "$BAKED_TAR" 2>/dev/null | awk '{print $5}')"
      else
        echo "[provision] WARNING: docker save failed"; rm -f "${BAKED_TAR}.tmp.$$" 2>/dev/null
      fi
    else
      echo "[provision] WARNING: docker commit failed"
    fi
  fi
  echo "[provision] DONE"
}

# =============================================================================
# PHASE: worker  (inside the container)
# =============================================================================
worker() {
  cd "$REPO_ROOT"
  mkdir -p "$SCRATCH" 2>/dev/null || true
  LOG=${LOG:-$SCRATCH/yambda_5b_8gpu.log}
  # Avoid double-logging. When launched by the orchestrate phase, our stdout is
  # ALREADY captured into the real $LOG by orchestrate's `tee` (and, multi-node,
  # funneled through one srun pipe). Re-`tee`ing $LOG here would write every line
  # twice. Orchestrate sets WORKER_TEE=0 to point our own file sink at /dev/null:
  # we still echo to stdout (captured upstream) but don't duplicate the file.
  # Direct single-node invocation (the streaming-e2e supervisor) leaves
  # WORKER_TEE unset, so the worker keeps writing $LOG itself.
  [ "${WORKER_TEE:-1}" = "0" ] && LOG=/dev/null
  # TensorBoard under the writable scratch root unless the caller (e.g. the e2e
  # supervisor) pinned a per-run path. Keeps the gin default from ever being used.
  export TENSORBOARD_LOG_PATH=${TENSORBOARD_LOG_PATH:-$SCRATCH/tb/yambda_5b}
  # MLPerf Training compliance log (streaming-train-eval path). Lands beside the
  # other run outputs under scratch unless the caller pins it. Rank 0 writes it;
  # check it post-run with:
  #   python -m mlperf_logging.compliance_checker --usage training \
  #     --ruleset 5.0.0 "$MLPERF_LOG_PATH"
  # Default to a PER-JOB filename so each standalone `sbatch` gets a clean
  # compliance log: mllog opens the file in APPEND mode, so a fixed name would
  # accumulate events across runs and fail the compliance_checker (duplicate
  # INIT_START/RUN_START). The streaming-e2e supervisor pins MLPERF_LOG_PATH
  # explicitly (and inits it once at run start), so its relaunch-into-same-file
  # append semantics are preserved untouched.
  export MLPERF_LOG_PATH=${MLPERF_LOG_PATH:-$SCRATCH/mlperf/yambda_5b_mlperf.${SLURM_JOB_ID:-manual}.log}
  echo "[$(date)] REPO_ROOT=$REPO_ROOT" | tee -a "$LOG"

  # polars-u64-idx (NOT stock polars) — yambda parquet's flat-explode overruns
  # 32-bit row index. Reserved node has no outbound DNS, so install from a
  # pre-staged tarball under /apps/chcai/. Override PIP_LOCAL_TGZ for other hosts.
  PIP_LOCAL_TGZ=${PIP_LOCAL_TGZ:-/apps/chcai/pip_local_yambda.tgz}   # [CLUSTER-SPECIFIC] shared-NFS path
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

  # --- distributed topology ---------------------------------------------------
  GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
  # Multi-node when launched one-task-per-node under SLURM (SLURM_NNODES>1);
  # otherwise fall through to legacy single-node defaults (localhost, node_rank 0).
  if [ "${SLURM_NNODES:-1}" -gt 1 ] && [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    NNODES=${SLURM_NNODES}
    NODE_RANK=${SLURM_NODEID:-${SLURM_PROCID:-0}}
    # PREFER a MASTER_ADDR/PORT forwarded from the orchestrate phase (resolved on
    # the host, which has scontrol); the container image carries no SLURM client.
    if [ -z "${MASTER_ADDR:-}" ]; then
      MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1)
      MASTER_ADDR=${MASTER_ADDR:-localhost}
    fi
    MASTER_PORT=${MASTER_PORT:-$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))}
  else
    NNODES=${NNODES:-1}
    NODE_RANK=${NODE_RANK:-0}
    # Single-node: all ranks live on THIS host, so rendezvous over loopback and
    # do NOT use the SLURM hostname. On some nodes the hostname resolves to a
    # non-routable per-GPU RoCE /31 (benic 192.168.x) address; using it makes the
    # NCCL bootstrap fail with "No route to host". localhost is node-independent.
    MASTER_ADDR=localhost
    MASTER_PORT=${MASTER_PORT:-}     # empty => train_ranker picks a free port
  fi
  export NNODES NODE_RANK GPUS_PER_NODE MASTER_ADDR MASTER_PORT
  export WORLD_SIZE=$(( NNODES * GPUS_PER_NODE ))
  echo "[$(date)] topology: nnodes=$NNODES node_rank=$NODE_RANK gpus_per_node=$GPUS_PER_NODE world_size=$WORLD_SIZE master=$MASTER_ADDR:${MASTER_PORT:-<auto>}" | tee -a "$LOG"

  # NCCL bootstrap NIC. The container is --network=host so RCCL sees ALL host
  # interfaces; if left to auto-detect, NCCL can pick a non-routable per-GPU RoCE
  # /31 (benic* 192.168.x) link and fail bootstrap with "No route to host" (this
  # is node-dependent: it worked on some nodes and not others, causing repetitive
  # single-node init failures). Pin it explicitly to avoid that.
  #   * Single-node (NNODES==1): all ranks are on THIS host, so only the bootstrap
  #     control-plane crosses the socket NIC (data plane is intra-node XGMI/PCIe,
  #     see below). Loopback is reachable by every local rank on ANY host and is
  #     node-independent — same rationale as MASTER_ADDR=localhost above — so it
  #     "just works" on dev boxes that have no fenic0 (e.g. a single MI355 node).
  #   * Multi-node (NNODES>1): needs a routable host NIC shared across nodes for
  #     the cross-node TCP rendezvous; default to the meta64 fenic0.
  # Both remain ${NCCL_SOCKET_IFNAME:-...}-overridable for other fabrics.
  # [CLUSTER-SPECIFIC] multi-node routable host NIC for TCP bootstrap (find via `ip -br addr`).
  if [ "$NNODES" -gt 1 ]; then
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-fenic0}
  else
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
  fi
  echo "[$(date)] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME (nnodes=$NNODES)" | tee -a "$LOG"

  # Multi-node additionally needs the RDMA data-plane (bnxt_re HCAs) configured;
  # single-node uses intra-node P2P (XGMI/PCIe) so only the bootstrap NIC matters.
  if [ "$NNODES" -gt 1 ]; then
    NCCL_NET_TRANSPORT=${NCCL_NET_TRANSPORT:-ib}
    if [ "$NCCL_NET_TRANSPORT" = "socket" ]; then
      export NCCL_IB_DISABLE=1
      echo "[$(date)] NCCL: IB disabled — allreduce over TCP (fenic0). Functional, not RDMA-fast." | tee -a "$LOG"
    else
      # bnxt_re userspace provider ABI overlay (REQUIRED for RCCL). The stock v34
      # provider faults RCCL's create_qp (256 WRs) against the host kernel uapi
      # ("Bad address"); the host v61/v59 set staged by the provision phase works.
      # The libibverbs.so (UNVERSIONED) symlink + LD_PRELOAD are both required so
      # the torch process maps ONLY the host lib (see provision phase comment).
      if [ -e "$OVERLAY/lib/libibverbs.so.1" ]; then
        [ -e "$OVERLAY/lib/libibverbs.so" ] || ln -sf libibverbs.so.1 "$OVERLAY/lib/libibverbs.so" 2>/dev/null || true
        export LD_LIBRARY_PATH="$OVERLAY/lib:$OVERLAY/lib/libibverbs:${LD_LIBRARY_PATH:-}"
        export LD_PRELOAD="$OVERLAY/lib/libibverbs.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "[$(date)] NCCL: bnxt_re provider overlay -> $OVERLAY (host rdma-core v61/v59; symlink+LD_PRELOAD so RCCL binds the host lib for QP creation)" | tee -a "$LOG"
      else
        echo "[$(date)] WARNING: RDMA overlay $OVERLAY missing — RCCL QP creation will fail 'Bad address' on stock v34 provider; set RDMA_OVERLAY or use NCCL_NET_TRANSPORT=socket" | tee -a "$LOG"
      fi
      # MINIMAL bnxt_re set PROVEN on these meta64 cv350 nodes (cmcknigh RCCL
      # benchmarks + confirmed e2e here). NCCL_IB_TC=104 (RoCE lossless PFC class)
      # is required; do NOT add the ionic-AINIC QPS/ECE/DMABUF block.
      # [CLUSTER-SPECIFIC] RDMA HCA names (`ibv_devices`); other vendors => mlx5_*/ionic_*
      export NCCL_IB_HCA=${NCCL_IB_HCA:-bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7}
      export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}    # [CLUSTER-SPECIFIC] RoCEv2 IPv4 GID idx (`show_gids`)
      export NCCL_IB_TC=${NCCL_IB_TC:-104}                # [CLUSTER-SPECIFIC] RoCE lossless/PFC traffic class
      export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-14}
      export NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY:-1}
      export RCCL_MSCCL_ENABLE=${RCCL_MSCCL_ENABLE:-0}
      # GPU-Direct RDMA: ENABLED by default. The brcmrdma host kernel ships the
      # inbox peer-memory client (`ib_register_peer_memory_client` in
      # /proc/kallsyms), so RCCL does true GPU<->NIC DMA over bnxt_re instead of
      # bouncing through host memory. Measured ~+22% throughput at 2 nodes
      # (65.7%->79.8% weak-scaling efficiency) vs the old host-staged path.
      # GDR_LEVEL=5 (most permissive) is required so GDR is used even when the GPU
      # and NIC cross the CPU root complex. NCCL_DMABUF_ENABLE=1 is a harmless
      # no-op here (kernel lacks CONFIG_DMABUF_MOVE_NOTIFY/CONFIG_PCI_P2PDMA, so
      # peermem carries it). Enabling is non-fatal: if peermem is ever absent RCCL
      # just logs "GDR 0" and falls back to host staging. Override with
      # NCCL_NET_GDR_LEVEL=0 to force the legacy host-staged path.
      export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-5}
      export NCCL_DMABUF_ENABLE=${NCCL_DMABUF_ENABLE:-1}
      echo "[$(date)] NCCL: RDMA over bnxt_re (GID idx ${NCCL_IB_GID_INDEX}, TC ${NCCL_IB_TC}, GDR_LEVEL=${NCCL_NET_GDR_LEVEL}, DMABUF=${NCCL_DMABUF_ENABLE}; meta64 bnxt_re config, validated)" | tee -a "$LOG"
    fi
  fi
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  export HSTU_HAMMER_KERNEL=${HSTU_HAMMER_KERNEL:-}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

  # --- GPU clock sanity guard -------------------------------------------------
  # A leftover perf_determinism cap (half clock) silently slows every kernel ~1.9x.
  # Log the perf level + a live sclk sample and try to restore boost (non-fatal).
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

  # --- stray-trainer / leaked-VRAM guard -------------------------------------
  # The trainer runs via `docker exec` into a long-lived container, so its procs
  # live in the container PID namespace, NOT the SLURM job cgroup. If a prior job
  # OOM'd/crashed, a rank can leak and keep holding ~half of every GPU's VRAM,
  # which persists across jobs (container survives) and guarantees the next
  # attempt OOMs. Before launching, reap any pre-existing trainer procs (there
  # should be none at this point) and wait for VRAM to drain. [g]-guard avoids
  # self-match. Non-fatal.
  if pgrep -f '[g]enerative_recommenders' >/dev/null 2>&1; then
    echo "[$(date)] WARNING: leaked trainer procs found pre-launch — killing." | tee -a "$LOG"
    pkill -9 -f '[g]enerative_recommenders' 2>/dev/null || true
    for _i in $(seq 1 15); do
      pgrep -f '[g]enerative_recommenders' >/dev/null 2>&1 || break
      sleep 2
    done
    sleep 5  # let the driver release VRAM after process exit
    if command -v rocm-smi >/dev/null 2>&1; then
      echo "[$(date)] post-cleanup GPU0 used GiB:$(rocm-smi --showmeminfo vram 2>/dev/null | awk -F: '/Used/{printf " %.0f", $3/1073741824; exit}')" | tee -a "$LOG"
    fi
  fi

  # WORKER_CMD override: run an arbitrary in-container command (e.g. an a2a/RCCL
  # micro-benchmark) instead of the trainer, REUSING all the NCCL/RDMA/topology
  # setup above so it exercises the exact transport the trainer uses. The
  # supervisor never sets WORKER_CMD, so the training path is unchanged.
  if [ -n "${WORKER_CMD:-}" ]; then
    echo "[$(date)] WORKER_CMD override (WORLD_SIZE=$WORLD_SIZE): $WORKER_CMD" | tee -a "$LOG"
    bash -lc "cd $REPO_ROOT && $WORKER_CMD" 2>&1 | tee -a "$LOG"
    return
  fi

  echo "[$(date)] launching train_ranker with WORLD_SIZE=$WORLD_SIZE" | tee -a "$LOG"
  python -m generative_recommenders.dlrm_v3.train.train_ranker \
      --dataset yambda-5b --mode "${MODE:-streaming-train-eval}" 2>&1 | tee -a "$LOG"
}

# ---- dispatch ---------------------------------------------------------------
case "$PHASE" in
  orchestrate) orchestrate ;;
  provision)   provision ;;
  worker)      worker ;;
  *) echo "launch_slurm.sh: unknown LAUNCH_SLURM_PHASE='$PHASE'" >&2; exit 2 ;;
esac
