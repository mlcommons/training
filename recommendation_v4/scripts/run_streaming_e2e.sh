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
#   ensure_ready -> acquire_node: submit an `sbatch` hold job (`--wrap "sleep
#   infinity"`, bounded by --time=$ALLOC_TIME) for a fresh exclusive node on
#   $PARTITION, optionally from --reservation $RESERVATION; wait for RUNNING,
#   then provision_node runs $PROVISION_SCRIPT on it via `srun --jobid --overlap`
#   (docker pull + container create + dep install; ~15 min on a cold node).
#   sbatch (not salloc) because interactive salloc on some partitions (e.g.
#   meta64) is capped at 240 min, which a multi-day hold would exceed. Jobs WE
#   create are tracked and `scancel`ed (container removed first) on success via
#   release_acquired; the user's original --jobid is never cancelled.
#   Checkpoints on shared NFS make the resume seamless.
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
#   failover:    --partition --reservation --alloc-time --allow-failover
#                --provision-script --acquire-wait-max --resv-wait-max
#                --orig-recover-wait
#                (failover holds <=1 reservation node: stray/leaked e2e_failover
#                 holds are reaped, and a lost ORIGINAL job is waited on for SLURM
#                 requeue and reused before a SEPARATE node is acquired.)
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
#       --jobid 12074 \
#       --ckpt-path /apps/chcai/ckpts/yambda_5b_e2e \
#       --run-name yambda_5b_e2e --log /apps/chcai/yambda_5b_e2e.log \
#       --start-ts 150 --num-train-ts 149 --eval-every 10 \
#       --ckpt-time-interval 3600 --keep-last-n 1 --max-relaunch 100 \
#       --reservation NAN_issue_debug \
#       > /apps/chcai/yambda_5b_e2e.supervisor.console.log 2>&1 &
#   (--reservation makes node-death failover re-acquire from that reservation;
#    omit it to fall back to the open $PARTITION pool.)
# =============================================================================

set -uo pipefail

JOBID=11367
CONTAINER=yambda_primus
REPO=/home/chcai/training/recommendation_v4

# Direct-SSH fallback so the supervisor can probe the node even while the SLURM
# control plane is unreachable — a transient controller outage must NOT be
# mistaken for node death (which would needlessly tear down a healthy run).
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
LAST_NODE=""   # last known node hostname for $JOBID (cached for direct probes)

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
KEEP_LAST_N=1
CKPT_PATH=/apps/chcai/ckpts/yambda_5b_e2e
RUN_NAME=yambda_5b_e2e
LOG=/apps/chcai/yambda_5b_e2e.log
MAX_RELAUNCH=50
NUM_TRAIN_BATCHES=0     # 0 = full window (only capped for validation/tests)
NUM_EVAL_BATCHES=0      # 0 = full holdout eval (only capped for validation)
DIE_AT_STEP=-1          # >=0 = test-only failure injection
# Train:eval split (fraction of USERS trained; 1 - this held out as a FIXED,
# never-trained eval set). Passed on EVERY relaunch so the split stays an
# immutable run contract — a changed split would abort on resume (validated in
# the loop) to prevent skip-offset desync and held-out users leaking into train.
TRAIN_SPLIT_PERCENTAGE=0.90
SPLIT_SALT=0
EVAL_HOLDOUT_TS=-1          # <0 = window just past training (start_ts+num_train_ts)
EVAL_HOLDOUT_NUM_WINDOWS=1
IN_WINDOW_FREQ=0        # >0 = also save every N batches within a window
ATTACH=0                # 1 = (re)attach to an already-running trainer without
                        #     killing it or truncating its log — used to restore
                        #     supervision over a trainer that outlived a previous
                        #     supervisor (e.g. one a control-plane outage killed).
CTRL_WAIT_MAX=3600      # max seconds to wait for an unreachable SLURM controller
                        # to recover before concluding failover is needed.

# --- node failover ----------------------------------------------------------
# If the current allocation/node goes away, acquire a FRESH node, (re)provision
# the container on it, and resume — checkpoints + code live on shared NFS
# (/apps/chcai, /home/chcai), so any node in the partition can continue.
PARTITION=meta64
RESERVATION=""                        # if set, failover acquires from this SLURM
                                      # reservation (e.g. NAN_issue_debug) so a
                                      # replacement node comes from the same pool.
ALLOC_TIME=7-00:00:00                 # SLURM --time for a failover hold job
ALLOW_FAILOVER=1                      # 0 = never acquire a new node
PROVISION_SCRIPT=/home/chcai/_provision_yambda_primus.sh
ACQUIRE_WAIT_MAX=1800                 # max seconds to wait for the OPEN-POOL
                                      # (tier-2) failover hold job to reach
                                      # RUNNING (tolerates brief queueing).
RESV_WAIT_MAX=300                     # max seconds to wait for a RESERVATION
                                      # (tier-1) node before giving up on it and
                                      # falling back to the open $PARTITION pool.
                                      # Short, since a free reservation node
                                      # starts ~immediately; a longer wait just
                                      # means the reservation is currently full.
ORIG_RECOVER_WAIT=600                 # when the user's ORIGINAL reservation job
                                      # is lost, wait this long for SLURM to
                                      # auto-requeue it back to RUNNING before
                                      # acquiring a SEPARATE node. Reusing the
                                      # requeued original keeps us at <=1
                                      # reservation node and skips a redundant
                                      # acquire (observed requeue latency ~2 min).

# Disk guard: require at least this many GiB free on the ckpt volume before a
# (re)launch. One checkpoint is ~560 GB. A save writes a fresh .tmp BEFORE the
# old copy is pruned, so peak transient usage is (keep_last_n + 1) copies. With
# keep_last_n=1 that is ~1120 GB; require ~1200 GiB free at launch so the run
# never wedges mid-save on a near-full shared NFS volume.
MIN_FREE_GIB=1200
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
        --attach) ATTACH="$2"; shift 2;;
        --ctrl-wait-max) CTRL_WAIT_MAX="$2"; shift 2;;
        --min-free-gib) MIN_FREE_GIB="$2"; shift 2;;
        --stall-s) STALL_S="$2"; shift 2;;
        --partition) PARTITION="$2"; shift 2;;
        --reservation) RESERVATION="$2"; shift 2;;
        --alloc-time) ALLOC_TIME="$2"; shift 2;;
        --allow-failover) ALLOW_FAILOVER="$2"; shift 2;;
        --provision-script) PROVISION_SCRIPT="$2"; shift 2;;
        --acquire-wait-max) ACQUIRE_WAIT_MAX="$2"; shift 2;;
        --resv-wait-max) RESV_WAIT_MAX="$2"; shift 2;;
        --orig-recover-wait) ORIG_RECOVER_WAIT="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

ORIGINAL_JOBID="$JOBID"   # never scancel the user's own hold allocation
ACQUIRED_JOBIDS=()        # failover allocations WE created (released on success)

SUP_LOG="${LOG%.log}.supervisor.log"

sup() { echo "[$(date '+%F %T')] [supervisor] $*" | tee -a "$SUP_LOG"; }

# Run a command inside the allocation's container, capturing its stdout. Wrapped
# in `timeout` so a hung control plane / NFS can never wedge the supervisor.
cexec() { timeout 90 srun --jobid="$JOBID" --overlap docker exec "$CONTAINER" bash -lc "$1" 2>/dev/null; }

# Is the SLURM control plane reachable right now?
controller_up() { timeout 12 sinfo -h -o '%P' >/dev/null 2>&1; }

# Refresh + echo the node hostname for $JOBID (cached in LAST_NODE for direct
# probes that must work even while the controller is down).
refresh_node() {
    local n; n=$(timeout 12 squeue -h -j "$JOBID" -o '%N' 2>/dev/null | head -1)
    [[ -n "$n" ]] && LAST_NODE="$n"
    echo "$LAST_NODE"
}

# Run a (simple) command in the container by SSHing the node DIRECTLY, bypassing
# SLURM — the only way to observe the trainer during a controller outage. Needs a
# previously-cached LAST_NODE. Keep "$1" free of embedded double quotes.
dexec() {
    [[ -z "$LAST_NODE" ]] && return 1
    timeout 40 ssh $SSH_OPTS "$LAST_NODE" "docker exec $CONTAINER bash -lc '$1'" 2>/dev/null
}

# Block (with backoff) until the controller is reachable again, up to
# CTRL_WAIT_MAX. A controller outage leaves RUNNING jobs running, so waiting it
# out is almost always preferable to abandoning a healthy node.
wait_for_controller() {
    local waited=0
    controller_up && return 0
    while ! controller_up; do
        if (( waited >= CTRL_WAIT_MAX )); then
            sup "controller still unreachable after ${waited}s (max ${CTRL_WAIT_MAX}s) — proceeding."
            return 1
        fi
        sup "SLURM controller unreachable; waiting for recovery (${waited}s/${CTRL_WAIT_MAX}s)…"
        sleep 30; waited=$((waited + 30))
    done
    sup "SLURM controller reachable again after ${waited}s."
    return 0
}

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
    timeout 30 srun --jobid="$1" --overlap docker exec "$CONTAINER" true >/dev/null 2>&1
}

# (Re)create + dep-install the container on the given allocation's node.
provision_node() {
    local jid="$1" node
    node=$(squeue -h -j "$jid" -o '%N' 2>/dev/null | head -1)
    sup "provisioning container '$CONTAINER' on job $jid (node ${node:-?}) — cold node can take ~15 min"
    srun --jobid="$jid" --overlap bash "$PROVISION_SCRIPT" >> "${LOG%.log}.provision.log" 2>&1
    container_up "$jid"
}

# Submit an sbatch hold job that merely pins one exclusive node (`sleep
# infinity`, bounded by --time=$ALLOC_TIME); echoes the jobid. $1 = extra sbatch
# args (e.g. "--reservation=NAN_issue_debug" or ""). sbatch (not salloc) because
# interactive salloc on some partitions (meta64) is capped at 240 min, which an
# $ALLOC_TIME multi-day hold exceeds. The container is provisioned afterward by
# provision_node via `srun --jobid --overlap`.
_submit_hold_job() {
    local extra="$1" out
    out=$(sbatch --parsable --partition="$PARTITION" $extra --nodes=1 --exclusive \
                 --time="$ALLOC_TIME" --job-name=e2e_failover \
                 --output="${LOG%.log}.failover_hold.%j.log" \
                 --wrap="echo \"[failover-hold] node=\$(hostname) jobid=\$SLURM_JOB_ID start=\$(date -Is)\"; sleep infinity" 2>&1)
    # --parsable => "<jobid>" or "<jobid>;<cluster>"; strip whitespace + cluster.
    echo "$out" | tr -d ' ' | cut -d';' -f1
}

# Wait up to $2 seconds for job $1 to reach RUNNING. Returns 0 if RUNNING.
_wait_running() {
    local jid="$1" max="$2" waited=0 st
    while (( waited < max )); do
        st=$(squeue -h -j "$jid" -o '%T' 2>/dev/null | head -1)
        [[ "$st" == "RUNNING" ]] && return 0
        sleep 10; waited=$((waited + 10))
    done
    return 1
}

# Acquire a fresh exclusive node and set global JOBID on success. Two-tier:
#   tier 1 (preferred): the SLURM --reservation $RESERVATION, if configured.
#     Waited on for only RESV_WAIT_MAX — a free reservation node starts almost
#     immediately, so a longer wait means the reservation is currently full.
#   tier 2 (fallback): the open $PARTITION pool (no reservation), waited on for
#     ACQUIRE_WAIT_MAX. Used when no reservation is set, or the reservation had
#     no node free within RESV_WAIT_MAX (the pending reservation job is
#     cancelled before we resubmit so we never end up holding two nodes).
acquire_node() {
    if [[ "$ALLOW_FAILOVER" != "1" ]]; then
        sup "failover disabled (--allow-failover 0); cannot acquire a new node"; return 1
    fi
    # Release any prior/leaked failover hold BEFORE grabbing a new one, so we
    # never transiently pin two reservation nodes (e.g. a dead tier-1 hold + the
    # replacement we are about to submit).
    reap_failover_holds ""
    local jid

    # --- tier 1: reservation (preferred) -------------------------------------
    if [[ -n "$RESERVATION" ]]; then
        sup "failover tier-1: requesting a node from reservation=$RESERVATION (exclusive, time=$ALLOC_TIME)"
        jid=$(_submit_hold_job "--reservation=$RESERVATION")
        if [[ "$jid" =~ ^[0-9]+$ ]]; then
            ACQUIRED_JOBIDS+=("$jid")   # track for cleanup even if it never starts
            sup "reservation hold job jobid=$jid submitted; waiting up to ${RESV_WAIT_MAX}s for RUNNING"
            if _wait_running "$jid" "$RESV_WAIT_MAX"; then
                JOBID="$jid"
                sup "new node ready (reservation $RESERVATION): jobid=$JOBID node=$(squeue -h -j "$JOBID" -o '%N' 2>/dev/null | head -1)"
                return 0
            fi
            sup "reservation $RESERVATION has no free node within ${RESV_WAIT_MAX}s — cancelling pending $jid and falling back to open pool"
            scancel "$jid" 2>/dev/null || true
        else
            sup "reservation sbatch did not return a jobid ($jid) — falling back to open pool"
        fi
    fi

    # --- tier 2: open partition pool (fallback) ------------------------------
    sup "failover tier-2: requesting a node from open partition=$PARTITION (exclusive, time=$ALLOC_TIME)"
    jid=$(_submit_hold_job "")
    if ! [[ "$jid" =~ ^[0-9]+$ ]]; then
        sup "FATAL: open-pool sbatch did not return a jobid: $jid"; return 1
    fi
    ACQUIRED_JOBIDS+=("$jid")
    sup "open-pool hold job jobid=$jid submitted; waiting up to ${ACQUIRE_WAIT_MAX}s for RUNNING"
    if _wait_running "$jid" "$ACQUIRE_WAIT_MAX"; then
        JOBID="$jid"
        sup "new node ready (open $PARTITION): jobid=$JOBID node=$(squeue -h -j "$JOBID" -o '%N' 2>/dev/null | head -1)"
        return 0
    fi
    sup "FATAL: open-pool hold job $jid never reached RUNNING (waited ${ACQUIRE_WAIT_MAX}s)"
    return 1
}

# Ensure $JOBID is a healthy allocation with the container up, failing over to a
# fresh provisioned node if not. Resume is automatic: the latest checkpoint is
# on shared NFS, reachable from whatever node we end up on.
ensure_ready() {
    # A controller outage leaves RUNNING jobs running; wait it out before deciding
    # anything is wrong, so we never abandon a healthy node over a transient blip.
    wait_for_controller || true
    if alloc_healthy "$JOBID"; then
        refresh_node >/dev/null
        if container_up "$JOBID"; then return 0; fi
        sup "alloc $JOBID healthy but container '$CONTAINER' not up — (re)provisioning"
        provision_node "$JOBID" && return 0
        sup "provisioning on $JOBID failed; will try a fresh node"
    else
        sup "current allocation $JOBID unavailable (job not RUNNING or node down/drained)"
        # Prefer the SLURM-requeued original over acquiring a SEPARATE node, so we
        # stay at <=1 reservation node. (No-op once we've already failed over off
        # the original.)
        if wait_for_original_recover; then
            JOBID="$ORIGINAL_JOBID"
            refresh_node >/dev/null
            if container_up "$JOBID"; then sup "reusing recovered original jobid=$JOBID"; return 0; fi
            sup "recovered original $JOBID up but container '$CONTAINER' not present — (re)provisioning"
            provision_node "$JOBID" && return 0
            sup "provisioning recovered original $JOBID failed; will acquire a fresh node"
        fi
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

# Enforce "at most ONE reservation node held by this run at a time" and reap
# orphans. Every node WE acquire is an `sbatch --job-name=e2e_failover` hold, so
# all our holds are discoverable by name even across a supervisor restart — which
# is how a previous supervisor that died mid-failover (e.g. on a provisioning
# error) can leave a hold pinning a second reservation node. Cancels every
# e2e_failover hold owned by us EXCEPT $1 (the one to keep) and the user's
# ORIGINAL_JOBID (never ours to cancel). Containers are removed before the node
# is freed so they don't linger for the next tenant.
reap_failover_holds() {
    local keep="${1:-}" me jid
    me=$(id -un 2>/dev/null)
    [[ -z "$me" ]] && return 0
    while read -r jid; do
        [[ -z "$jid" ]] && continue
        [[ "$jid" == "$keep" || "$jid" == "$ORIGINAL_JOBID" ]] && continue
        sup "reaping stray failover hold $jid (enforcing <=1 reservation node held by this run)"
        srun --jobid="$jid" --overlap docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        scancel "$jid" 2>/dev/null || true
    done < <(squeue -h -u "$me" -n e2e_failover -o '%i' 2>/dev/null)
}

# When the user's ORIGINAL reservation job is lost, SLURM typically auto-requeues
# it back onto a (fresh) reservation node within a couple of minutes. Waiting for
# that and REUSING it — rather than immediately acquiring a SEPARATE node — is
# what keeps us at <=1 reservation node (the alternative is the original requeue
# AND a failover hold both pinning reservation nodes) and skips a redundant
# acquire+provision. Only meaningful while we are still on the original job.
wait_for_original_recover() {
    [[ "$JOBID" != "$ORIGINAL_JOBID" ]] && return 1
    local waited=0
    while (( waited < ORIG_RECOVER_WAIT )); do
        if alloc_healthy "$ORIGINAL_JOBID"; then
            sup "original job $ORIGINAL_JOBID is RUNNING again (SLURM requeue) after ${waited}s — reusing it (no second node)"
            return 0
        fi
        sup "waiting for original job $ORIGINAL_JOBID to requeue before acquiring a separate node (${waited}s/${ORIG_RECOVER_WAIT}s)…"
        sleep 15; waited=$((waited + 15))
    done
    sup "original job $ORIGINAL_JOBID did not recover within ${ORIG_RECOVER_WAIT}s — acquiring a fresh node"
    return 1
}

# Returns 0 (true) if a trainer process is alive in the container. Uses SLURM
# (srun) when the controller is up, else falls back to a direct SSH probe so a
# control-plane outage can't make a live trainer look dead.
trainer_alive() {
    local n
    # `set -f; pgrep -f [g]enerative...` is the classic self-match guard: the
    # probe shell's OWN cmdline contains the pattern, so a naive `pgrep -f
    # generative_recommenders` ALWAYS matches itself and returns >=1 even when
    # the trainer is dead — which would defeat the stall watchdog and make
    # ATTACH mode falsely "adopt" a nonexistent trainer. The [g] char-class
    # matches "generative" in real trainer cmdlines but NOT the literal
    # "[g]enerative" in the probe's cmdline; `set -f` keeps the bracket from
    # being glob-expanded (works under both bash -lc wrappers, no quotes).
    if controller_up; then
        n=$(cexec "set -f; pgrep -f [g]enerative_recommenders | wc -l" | tr -d ' ')
    else
        n=$(dexec "set -f; pgrep -f [g]enerative_recommenders | wc -l" | tr -d ' ')
    fi
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
        TRAIN_SPLIT_PERCENTAGE=$TRAIN_SPLIT_PERCENTAGE \
        SPLIT_SALT=$SPLIT_SALT \
        EVAL_HOLDOUT_TS=$EVAL_HOLDOUT_TS \
        EVAL_HOLDOUT_NUM_WINDOWS=$EVAL_HOLDOUT_NUM_WINDOWS \
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
sup "failover: allow=$ALLOW_FAILOVER partition=$PARTITION reservation=${RESERVATION:-<none>} alloc_time=$ALLOC_TIME"

# Reap any failover hold(s) leaked by a PREVIOUS supervisor that died mid-failover
# (e.g. exited on a provisioning error before release_acquired could run). Without
# this, such an orphan keeps pinning a second reservation node indefinitely.
reap_failover_holds ""

cexec "mkdir -p '$CKPT_PATH' '/apps/chcai/tb/$RUN_NAME'"
# Initialize this run's metrics log ONCE. launch_smoke_8gpu.sh appends (tee -a),
# so every relaunch attempt accumulates into this single file — the full-run
# NE/AUC history survives crashes and node failover instead of being truncated
# on each relaunch. (Starting the supervisor = starting a fresh run.) In ATTACH
# mode we are adopting an already-running trainer, so we KEEP its existing log.
if [[ "$ATTACH" == "1" ]]; then
    sup "ATTACH mode: adopting existing run — keeping metrics log intact: $LOG"
else
    cexec ": > '$LOG'"
    sup "metrics log initialized (relaunch-append): $LOG"
fi
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
    refresh_node >/dev/null   # cache LAST_NODE for direct probes during outages

    # ATTACH (first attempt only): if a trainer is already running for this run,
    # adopt it in place — DON'T disk-guard (its sweep would delete an in-flight
    # .tmp save), DON'T cleanup_workers (would kill it), DON'T launch. Just begin
    # monitoring. Any subsequent relaunch is a normal launch from the checkpoint.
    adopt=0
    if [[ "$ATTACH" == "1" ]] && trainer_alive; then
        adopt=1; ATTACH=0
        sup "ATTACH mode: trainer already alive on ${LAST_NODE:-node} — monitoring in place (no relaunch/kill/sweep)."
    fi

    if (( adopt )); then
        # Mark current end of log so we only read sentinels produced from here on.
        start_line=$(cexec "wc -l < '$LOG' 2>/dev/null" | tr -d ' '); start_line=${start_line:-0}
        start_line=$((start_line + 1))
        sup "monitoring adopted run (reading sentinels from log line $start_line)"
    else
        if ! disk_guard; then exit 3; fi
        cleanup_workers
        # Mark current end of log so we only read sentinels produced by THIS attempt.
        start_line=$(cexec "wc -l < '$LOG' 2>/dev/null" | tr -d ' '); start_line=${start_line:-0}
        start_line=$((start_line + 1))
        sup "launching (reading sentinels from log line $start_line)"
        launch
        sleep 15  # let docker exec spin up the process
    fi

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
            if ! controller_up; then
                # Control plane unreachable != node down. If the trainer is still
                # alive on the node (direct SSH probe), this is a transient blip —
                # keep monitoring rather than tearing down a healthy run.
                if trainer_alive; then
                    sup "control plane unreachable but trainer still alive on ${LAST_NODE:-node} — transient; continuing to monitor."
                else
                    sup "control plane unreachable AND trainer absent on ${LAST_NODE:-node} — relaunching with failover."
                    break
                fi
            else
                sup "allocation $JOBID lost mid-run (node down/job ended) — relaunching with failover."
                break
            fi
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
