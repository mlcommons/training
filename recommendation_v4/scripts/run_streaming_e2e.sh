#!/bin/bash
# =============================================================================
# run_streaming_e2e.sh — self-healing supervisor for the long-run yambda-5b
#                        streaming train+eval (NE/AUC over the full ~5B dataset)
# =============================================================================
#
# WHAT IT DOES
#   Owns a multi-day "streaming-train-eval" run and keeps it alive unattended
#   across the three failure modes that actually kill long runs:
#       1. trainer process crash / OOM / nonzero exit
#       2. silent death (the whole process group gets SIGKILLed — no exit code)
#       3. the SLURM node itself going away (down / drained / job ended)
#   In every case it relaunches the trainer from the latest on-disk checkpoint
#   (failing over to a brand-new node for case 3) until the run finishes.
#
# WHY A RELAUNCH "JUST WORKS" (resume model)
#   The training stack already implements exact-once resume: on startup it picks
#   the latest numeric checkpoint subdir under $CKPT_PATH, restores model +
#   optimizer + per-rank RNG, and (for mid-window in-window saves) skips the
#   batches already trained in the partially-done window. So relaunching with
#   the SAME --ckpt-path transparently continues from where it died — no manual
#   bookkeeping here beyond pointing every attempt at the same base dir.
#
# WHERE IT RUNS / HOW IT DRIVES WORK
#   This script runs on the SLURM HEAD node. The trainer runs inside a long-
#   lived docker container ($CONTAINER) on the compute node held by a SLURM
#   allocation ($JOBID). All control flow is `srun --jobid <id> --overlap
#   docker exec ...` into that container. The container bind-mounts shared NFS
#   (/home/chcai = code, /apps/chcai = checkpoints+logs), which is what makes
#   node failover possible: any node in $PARTITION sees the same code+state.
#
# MAIN LOOP (state machine, up to --max-relaunch attempts)
#   for each attempt:
#     ensure_ready   — guarantee a healthy allocation whose container is up,
#                      failing over to a freshly-provisioned node if not.
#     disk_guard     — sweep crash-orphaned *.tmp/*.old saves; abort if the
#                      ckpt volume has < --min-free-gib free.
#     cleanup_workers— kill any stragglers from a previous attempt.
#     launch         — detached `docker exec -d` of the trainer; a trailing
#                      echo appends an `E2E_RUN_EXIT=<code>` sentinel to the log
#                      when the trainer returns (clean OR crash).
#     monitor loop (every --poll-s):
#         * node watchdog  — if $JOBID stops being healthy mid-run, break and
#                            let the next attempt fail over.
#         * exit sentinel  — E2E_RUN_EXIT=0 => success (done); nonzero => relaunch.
#         * stall watchdog — if the log stops growing AND no trainer process is
#                            alive for --stall-s, treat as silent death=>relaunch.
#                            (Long blocking saves keep the process alive, so they
#                            never false-trip this.)
#
# NODE FAILOVER (case 3, the --allow-failover path)
#   ensure_ready -> acquire_node: `salloc --no-shell --exclusive` a fresh node on
#   $PARTITION, wait for RUNNING, then provision_node runs $PROVISION_SCRIPT on
#   it (docker pull + container create + dep install; ~15 min on a cold node).
#   Allocations WE create are tracked and `scancel`ed (container removed first)
#   on success via release_acquired; the user's original --jobid is never
#   cancelled. Checkpoints on shared NFS make the resume seamless.
#
# CHECKPOINTS / DISK
#   The trainer saves atomically (write to <ts>.tmp, fsync, rename to <ts>) and
#   prunes to keep_last_n newest. One checkpoint is ~560 GB; a save blocks the
#   step it fires on for ~83 s (measured, no NFS contention). Cadence is driven
#   by --ckpt-time-interval (time-based) and optional --in-window-freq.
#
# ARGS (all optional; defaults target the full production run)
#   run shape:   --jobid --container --start-ts --num-train-ts --eval-every
#   ckpt:        --ckpt-path --keep-last-n --ckpt-time-interval --in-window-freq
#   logging:     --run-name --log
#   resilience:  --max-relaunch --min-free-gib --stall-s
#   failover:    --partition --alloc-time --allow-failover --provision-script
#   validation:  --num-train-batches --num-eval-batches  (>0 caps batches/window
#                for fast tests; 0 = full window / full-holdout eval)
#   test-only:   --die-at-step  (>=0 injects a crash at that global step)
#
# EXIT CODES
#   0  run completed (E2E_RUN_EXIT=0 — all windows + final eval done)
#   1  exhausted --max-relaunch without completing
#   3  disk guard tripped (insufficient free space)
#   4  could not secure a healthy allocation (failover failed / disabled)
#
# OUTPUTS (next to --log)
#   <log>                      trainer stdout/stderr + E2E_RUN_EXIT sentinels
#   <log:.log>.supervisor.log  this supervisor's own timeline
#   <log:.log>.provision.log   node-provisioning output (failover only)
#
# EXAMPLE
#   nohup bash scripts/run_streaming_e2e.sh \
#       --ckpt-path /apps/chcai/ckpts/yambda_5b_e2e \
#       --run-name yambda_5b_e2e --log /apps/chcai/yambda_5b_e2e.log \
#       --start-ts 150 --num-train-ts 149 --eval-every 10 \
#       --ckpt-time-interval 7200 --keep-last-n 2 --max-relaunch 50 \
#       > /apps/chcai/yambda_5b_e2e.supervisor.console.log 2>&1 &
# =============================================================================

set -uo pipefail

JOBID=11367
CONTAINER=yambda_primus
REPO=/home/chcai/training/recommendation_v4

# Defaults are sized from measurement: ~560 GB/checkpoint, ~83 s/save (blocking,
# attributed to the step it fires on), ~650 ms/train step @ global batch 8192,
# ~1465 steps (~16 min) per full ~12M-anchor window, full-holdout eval
# ~6-7 min/window. A ~2h time-based checkpoint interval keeps save overhead ~1%
# while bounding crash-loss to ~2h of compute; eval every N windows
# (EVAL_EVERY_N_WINDOWS) amortizes the full-holdout eval cost.
NUM_TRAIN_TS=149
START_TS=150
EVAL_EVERY=5
CKPT_TIME_INTERVAL=7200
KEEP_LAST_N=2
CKPT_PATH=/apps/chcai/ckpts/yambda_5b_e2e
RUN_NAME=yambda_5b_e2e
LOG=/apps/chcai/yambda_5b_e2e.log
MAX_RELAUNCH=50
NUM_TRAIN_BATCHES=0     # 0 = full window (only capped for validation/tests)
NUM_EVAL_BATCHES=0      # 0 = full holdout eval (only capped for validation)
DIE_AT_STEP=-1          # >=0 = test-only failure injection
IN_WINDOW_FREQ=0        # >0 = also save every N batches within a window

# --- node failover ----------------------------------------------------------
# If the current allocation/node goes away, acquire a FRESH node, (re)provision
# the container on it, and resume — checkpoints + code live on shared NFS
# (/apps/chcai, /home/chcai), so any node in the partition can continue.
PARTITION=meta64
ALLOC_TIME=7-00:00:00                 # SLURM --time for a failover allocation
ALLOW_FAILOVER=1                      # 0 = never acquire a new node
PROVISION_SCRIPT=/home/chcai/_provision_yambda_primus.sh

# Disk guard: require at least this many GiB free on the ckpt volume before a
# (re)launch. One checkpoint is ~600 GB; with keep_last_n the existing copies
# are already counted as used, so we only need room for one new in-flight .tmp
# plus margin (~800 GiB). The volume has ~3.7 TB free.
MIN_FREE_GIB=800
# Stall watchdog: if the log hasn't grown AND no trainer process is alive for
# this many seconds with no exit sentinel, treat it as a silent death. Comfortably
# exceeds one blocking checkpoint save (~83 s); and because a save keeps the
# trainer process alive, an in-progress save never trips the watchdog anyway.
STALL_S=1200
POLL_S=30

while [[ $# -gt 0 ]]; do
    case $1 in
        --jobid) JOBID="$2"; shift 2;;
        --container) CONTAINER="$2"; shift 2;;
        --num-train-ts) NUM_TRAIN_TS="$2"; shift 2;;
        --start-ts) START_TS="$2"; shift 2;;
        --eval-every) EVAL_EVERY="$2"; shift 2;;
        --ckpt-time-interval) CKPT_TIME_INTERVAL="$2"; shift 2;;
        --keep-last-n) KEEP_LAST_N="$2"; shift 2;;
        --ckpt-path) CKPT_PATH="$2"; shift 2;;
        --run-name) RUN_NAME="$2"; shift 2;;
        --log) LOG="$2"; shift 2;;
        --max-relaunch) MAX_RELAUNCH="$2"; shift 2;;
        --num-train-batches) NUM_TRAIN_BATCHES="$2"; shift 2;;
        --num-eval-batches) NUM_EVAL_BATCHES="$2"; shift 2;;
        --die-at-step) DIE_AT_STEP="$2"; shift 2;;
        --in-window-freq) IN_WINDOW_FREQ="$2"; shift 2;;
        --min-free-gib) MIN_FREE_GIB="$2"; shift 2;;
        --stall-s) STALL_S="$2"; shift 2;;
        --partition) PARTITION="$2"; shift 2;;
        --alloc-time) ALLOC_TIME="$2"; shift 2;;
        --allow-failover) ALLOW_FAILOVER="$2"; shift 2;;
        --provision-script) PROVISION_SCRIPT="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

ORIGINAL_JOBID="$JOBID"   # never scancel the user's own hold allocation
ACQUIRED_JOBIDS=()        # failover allocations WE created (released on success)

SUP_LOG="${LOG%.log}.supervisor.log"

sup() { echo "[$(date '+%F %T')] [supervisor] $*" | tee -a "$SUP_LOG"; }

# Run a command inside the allocation's container, capturing its stdout.
cexec() { srun --jobid="$JOBID" --overlap docker exec "$CONTAINER" bash -lc "$1" 2>/dev/null; }

cleanup_workers() {
    # The trainer spawns 8 rank processes + dataloader workers whose cmdlines
    # don't all match `train_ranker`/`spawn_main`, so target them, then fall
    # back to `pkill python` — safe because this container is dedicated to this
    # training (only the trainer runs python here during a supervised run).
    srun --jobid="$JOBID" --overlap docker exec "$CONTAINER" bash -lc \
        "pkill -9 -f train_ranker 2>/dev/null; pkill -9 -f multiprocessing 2>/dev/null; \
         sleep 2; pkill -9 python 2>/dev/null; sleep 3; true" 2>/dev/null || true
}

# --- node-failover helpers ---------------------------------------------------

# Healthy = the job is RUNNING and its node is not down/drained/failing.
alloc_healthy() {
    local jid="$1"
    [[ -z "$jid" ]] && return 1
    local st node nstate
    st=$(squeue -h -j "$jid" -o '%T' 2>/dev/null | head -1)
    [[ "$st" != "RUNNING" ]] && return 1
    node=$(squeue -h -j "$jid" -o '%N' 2>/dev/null | head -1)
    [[ -z "$node" ]] && return 1
    nstate=$(sinfo -h -n "$node" -o '%t' 2>/dev/null | head -1)
    case "$nstate" in
        *down*|*drain*|*fail*|*unk*|*boot*|"") return 1;;
    esac
    return 0
}

# Can we actually exec in the training container on this allocation?
container_up() {
    srun --jobid="$1" --overlap docker exec "$CONTAINER" true >/dev/null 2>&1
}

# (Re)create + dep-install the container on the given allocation's node.
provision_node() {
    local jid="$1" node
    node=$(squeue -h -j "$jid" -o '%N' 2>/dev/null | head -1)
    sup "provisioning container '$CONTAINER' on job $jid (node ${node:-?}) — cold node can take ~15 min"
    srun --jobid="$jid" --overlap bash "$PROVISION_SCRIPT" >> "${LOG%.log}.provision.log" 2>&1
    container_up "$jid"
}

# Acquire a fresh exclusive node on $PARTITION; sets global JOBID on success.
acquire_node() {
    if [[ "$ALLOW_FAILOVER" != "1" ]]; then
        sup "failover disabled (--allow-failover 0); cannot acquire a new node"; return 1
    fi
    sup "requesting a fresh node on partition=$PARTITION (exclusive, time=$ALLOC_TIME)"
    local out jid
    out=$(salloc --no-shell --partition="$PARTITION" --nodes=1 --exclusive \
                 --time="$ALLOC_TIME" --job-name=e2e_failover 2>&1)
    jid=$(echo "$out" | grep -oiE "Granted job allocation [0-9]+" | grep -oE "[0-9]+" | head -1)
    if [[ -z "$jid" ]]; then
        sup "FATAL: salloc did not grant a node: $out"; return 1
    fi
    ACQUIRED_JOBIDS+=("$jid")
    sup "granted new allocation jobid=$jid; waiting for RUNNING"
    local waited=0
    while (( waited < 600 )); do
        [[ "$(squeue -h -j "$jid" -o '%T' 2>/dev/null | head -1)" == "RUNNING" ]] && break
        sleep 10; waited=$((waited + 10))
    done
    if [[ "$(squeue -h -j "$jid" -o '%T' 2>/dev/null | head -1)" != "RUNNING" ]]; then
        sup "FATAL: new allocation $jid never reached RUNNING (waited ${waited}s)"; return 1
    fi
    JOBID="$jid"
    sup "new node ready: jobid=$JOBID node=$(squeue -h -j "$JOBID" -o '%N' 2>/dev/null | head -1)"
    return 0
}

# Ensure $JOBID is a healthy allocation with the container up, failing over to a
# fresh provisioned node if not. Resume is automatic: the latest checkpoint is
# on shared NFS, reachable from whatever node we end up on.
ensure_ready() {
    if alloc_healthy "$JOBID"; then
        if container_up "$JOBID"; then return 0; fi
        sup "alloc $JOBID healthy but container '$CONTAINER' not up — (re)provisioning"
        provision_node "$JOBID" && return 0
        sup "provisioning on $JOBID failed; will try a fresh node"
    else
        sup "current allocation $JOBID unavailable (job not RUNNING or node down/drained)"
    fi
    acquire_node || return 1
    provision_node "$JOBID" || { sup "provisioning new node $JOBID failed"; return 1; }
    sup "failover complete — now running on jobid=$JOBID"
    return 0
}

release_acquired() {
    local jid
    for jid in "${ACQUIRED_JOBIDS[@]:-}"; do
        [[ -n "$jid" && "$jid" != "$ORIGINAL_JOBID" ]] || continue
        # docker is independent of SLURM, so remove the container before freeing
        # the node, otherwise it lingers for the next tenant.
        srun --jobid="$jid" --overlap docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        scancel "$jid" 2>/dev/null && sup "released failover allocation $jid (container removed)"
    done
}

# Returns 0 (true) if a trainer process is alive in the container.
trainer_alive() {
    local n
    n=$(cexec "pgrep -f generative_recommenders | wc -l" | tr -d ' ')
    [[ "${n:-0}" -gt 0 ]]
}

disk_guard() {
    # Sweep crash-orphaned partial saves, then check free space.
    cexec "for d in '$CKPT_PATH'/*.tmp '$CKPT_PATH'/*.old; do [ -e \"\$d\" ] && rm -rf \"\$d\" && echo swept \"\$d\"; done; true"
    local free_gib
    free_gib=$(cexec "df -BG --output=avail '$CKPT_PATH' 2>/dev/null | tail -1 | tr -dc '0-9'")
    free_gib=${free_gib:-0}
    sup "disk guard: ${free_gib} GiB free on $CKPT_PATH (min ${MIN_FREE_GIB})"
    if (( free_gib < MIN_FREE_GIB )); then
        sup "FATAL: insufficient free space (${free_gib} < ${MIN_FREE_GIB} GiB). Aborting."
        return 1
    fi
    return 0
}

launch() {
    # Detached launch. The trailing echo appends a definitive exit sentinel to
    # the log once the trainer returns (clean finish OR crash with nonzero rc).
    srun --jobid="$JOBID" --overlap docker exec -d "$CONTAINER" bash -lc "
        cd $REPO &&
        HSTU_HAMMER_KERNEL=TRITON \
        MODE=streaming-train-eval \
        START_TS=$START_TS \
        NUM_TRAIN_TS=$NUM_TRAIN_TS \
        EVAL_EACH_WINDOW=1 \
        EVAL_EVERY_N_WINDOWS=$EVAL_EVERY \
        CKPT_PATH=$CKPT_PATH \
        KEEP_LAST_N=$KEEP_LAST_N \
        CKPT_TIME_INTERVAL_S=$CKPT_TIME_INTERVAL \
        IN_WINDOW_CKPT_FREQ=$IN_WINDOW_FREQ \
        NUM_TRAIN_BATCHES=$NUM_TRAIN_BATCHES \
        NUM_EVAL_BATCHES=$NUM_EVAL_BATCHES \
        DIE_AT_STEP=$DIE_AT_STEP \
        METRIC_LOG_FREQ=50 \
        RUN_NAME=$RUN_NAME \
        TENSORBOARD_LOG_PATH=/apps/chcai/tb/$RUN_NAME/ \
        LOG=$LOG \
        bash scripts/launch_smoke_8gpu.sh;
        echo \"E2E_RUN_EXIT=\$? \$(date '+%F %T')\" >> $LOG
    "
}

# Returns the exit code from the most recent E2E_RUN_EXIT sentinel APPENDED
# since `since_marker` bytes, or empty if none yet.
last_exit_since() {
    local since_line="$1"
    cexec "tail -n +$since_line '$LOG' 2>/dev/null | grep -aoE 'E2E_RUN_EXIT=[0-9]+' | tail -1 | cut -d= -f2"
}

sup "=== streaming e2e supervisor start ==="
sup "jobid=$JOBID container=$CONTAINER repo=$REPO"
sup "start_ts=$START_TS num_train_ts=$NUM_TRAIN_TS eval_every=$EVAL_EVERY"
sup "ckpt_path=$CKPT_PATH keep_last_n=$KEEP_LAST_N ckpt_time_interval=${CKPT_TIME_INTERVAL}s in_window_freq=$IN_WINDOW_FREQ"
sup "log=$LOG num_train_batches=$NUM_TRAIN_BATCHES die_at_step=$DIE_AT_STEP max_relaunch=$MAX_RELAUNCH"

cexec "mkdir -p '$CKPT_PATH' '/apps/chcai/tb/$RUN_NAME'"
# Initialize this run's metrics log ONCE. launch_smoke_8gpu.sh appends (tee -a),
# so every relaunch attempt accumulates into this single file — the full-run
# NE/AUC history survives crashes and node failover instead of being truncated
# on each relaunch. (Starting the supervisor = starting a fresh run.)
cexec ": > '$LOG'"
sup "metrics log initialized (relaunch-append): $LOG"
sup "tensorboard (NFS): /apps/chcai/tb/$RUN_NAME/"

attempt=0
while (( attempt < MAX_RELAUNCH )); do
    attempt=$((attempt + 1))
    sup "--- attempt $attempt/$MAX_RELAUNCH ---"

    # Make sure we have a live, container-ready node (failover + provision if the
    # current allocation/node has gone away).
    if ! ensure_ready; then
        sup "FATAL: could not secure a healthy allocation (failover failed)."
        exit 4
    fi
    if ! disk_guard; then exit 3; fi
    cleanup_workers

    # Mark current end of log so we only read sentinels produced by THIS attempt.
    start_line=$(cexec "wc -l < '$LOG' 2>/dev/null" | tr -d ' '); start_line=${start_line:-0}
    start_line=$((start_line + 1))

    sup "launching (reading sentinels from log line $start_line)"
    launch
    sleep 15  # let docker exec spin up the process

    # Monitor loop.
    last_size=0
    stall_accum=0
    hb=0
    while true; do
        # Node/allocation watchdog: if the node we're on goes down/drains or the
        # job ends, bail out of the monitor — the next attempt's ensure_ready
        # will fail over to a fresh node and resume from the latest checkpoint.
        hb=$((hb + 1))
        if (( hb % 4 == 0 )) && ! alloc_healthy "$JOBID"; then
            sup "allocation $JOBID lost mid-run (node down/job ended) — relaunching with failover."
            break
        fi

        rc=$(last_exit_since "$start_line")
        if [[ -n "$rc" ]]; then
            if [[ "$rc" == "0" ]]; then
                sup "RUN COMPLETED CLEANLY (E2E_RUN_EXIT=0) on attempt $attempt."
                cleanup_workers
                final_ckpts=$(cexec "ls '$CKPT_PATH' 2>/dev/null | grep -E '^[0-9]+$' | tr '\n' ' '")
                sup "final checkpoints retained: ${final_ckpts:-<none>}"
                release_acquired
                sup "=== streaming e2e supervisor done (success) ==="
                exit 0
            fi
            sup "trainer exited nonzero (E2E_RUN_EXIT=$rc). Will relaunch from latest checkpoint."
            break
        fi

        # Stall watchdog: track log growth; if frozen and no trainer alive, die.
        cur_size=$(cexec "wc -c < '$LOG' 2>/dev/null" | tr -d ' '); cur_size=${cur_size:-0}
        if [[ "$cur_size" == "$last_size" ]]; then
            if trainer_alive; then
                stall_accum=0  # alive but quiet (e.g. long save / eval) — ok
            else
                stall_accum=$((stall_accum + POLL_S))
                if (( stall_accum >= STALL_S )); then
                    sup "STALL: log frozen ${stall_accum}s and no trainer alive — silent death. Relaunching."
                    break
                fi
            fi
        else
            stall_accum=0
            last_size=$cur_size
        fi
        sleep "$POLL_S"
    done

    cleanup_workers
    sleep $(( attempt < 5 ? 20 : 60 ))  # small backoff
done

sup "FATAL: exhausted MAX_RELAUNCH=$MAX_RELAUNCH without completion."
sup "=== streaming e2e supervisor done (failure) ==="
exit 1
