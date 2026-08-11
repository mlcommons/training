#!/bin/bash
# =============================================================================
# run_streaming_e2e_local.sh — self-healing supervisor for a SINGLE-HOST
#   (NON-SLURM) yambda-5b streaming train+eval run. Local analog of
#   scripts/run_streaming_e2e.sh (the SLURM/sbatch supervisor).
#
# WHAT IT SUPERVISES
#   The "job" is one foreground run of --submit-script (default
#   scripts/launch_e2e_local.sh), which `docker exec`s the trainer in the
#   container. The supervisor runs that submit-script in the BACKGROUND so:
#     * its host PID == liveness  (kill -0 / hang watchdog), and
#     * `wait $PID` == the trainer's EXIT CODE (success vs. failure).
#   On crash / nonzero-exit / hang it RELAUNCHES the same submit-script (same
#   $CKPT_PATH/$LOG → resumes from the latest checkpoint), bounded by
#   --max-relaunch. This is the SLURM supervisor's sacct/squeue/scancel control
#   plane re-expressed with a local process + `docker exec` lifecycle.
#
# WHAT IT DETECTS (poll every --poll-s)
#   * submit-script process exits -> read its exit code:
#       0      => run finished cleanly (success)
#       != 0   => crash/OOM/die_at_step(42)/etc. => relaunch from latest ckpt
#   * hang watchdog: process alive but $LOG frozen >= --stall-s AND no trainer
#       process alive in the container (pgrep via docker exec) => kill + pkill in
#       container + relaunch. A long eval / blocking ckpt save keeps the trainer
#       process up, so it is NOT counted as a stall.
#   * disk guard before each (re)launch: require --min-free-gib on the ckpt vol.
#
# USAGE
#   nohup bash scripts/run_streaming_e2e_local.sh \
#       --submit-script scripts/launch_e2e_local.sh \
#       --log       /home/chcai/yambda_5b_e2e/<run>/<run>.log \
#       --ckpt-path /home/chcai/yambda_5b_e2e/<run>/ckpts \
#       --run-name  <run> \
#       > /home/chcai/yambda_5b_e2e/<run>/<run>.supervisor.console.log 2>&1 &
#
#   Per-run hyperparameters live in the --submit-script's env defaults (or are
#   exported before invoking this supervisor), not here.
#
# EXIT CODES
#   0  run completed cleanly
#   1  exhausted --max-relaunch without completion (or launch failed)
#   3  disk guard tripped
# =============================================================================
set -uo pipefail

SUBMIT_SCRIPT="scripts/launch_e2e_local.sh"
LOG=""
CKPT_PATH=""
RUN_NAME="yambda_5b_e2e_local"
CONTAINER=${CONTAINER:-yambda_local}
DOCKER=${DOCKER:-sudo docker}
MAX_RELAUNCH=50
MIN_FREE_GIB=700        # one full DMP ckpt (~600 GB) + headroom for the atomic
                        # .tmp written beside the retained one during a save.
STALL_S=2400            # 40 min frozen-log + no-trainer-proc => hung.
POLL_S=30

while [[ $# -gt 0 ]]; do
    case $1 in
        --submit-script) SUBMIT_SCRIPT="$2"; shift 2;;
        --log) LOG="$2"; shift 2;;
        --ckpt-path) CKPT_PATH="$2"; shift 2;;
        --run-name) RUN_NAME="$2"; shift 2;;
        --container) CONTAINER="$2"; shift 2;;
        --docker) DOCKER="$2"; shift 2;;
        --max-relaunch) MAX_RELAUNCH="$2"; shift 2;;
        --min-free-gib) MIN_FREE_GIB="$2"; shift 2;;
        --stall-s) STALL_S="$2"; shift 2;;
        --poll-s) POLL_S="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

[[ -n "$SUBMIT_SCRIPT" && -f "$SUBMIT_SCRIPT" ]] || { echo "FATAL: --submit-script required and must exist ($SUBMIT_SCRIPT)"; exit 1; }
[[ -n "$LOG" ]] || { echo "FATAL: --log required"; exit 1; }

SUP_LOG="${LOG%.log}.supervisor.log"
# Create the log + ckpt dirs up front so the disk guard's df has a real path to
# stat (df on a nonexistent dir returns 0 avail -> false "disk full" abort).
mkdir -p "$(dirname "$SUP_LOG")"
[[ -n "$CKPT_PATH" ]] && mkdir -p "$CKPT_PATH"
sup() { echo "[$(date '+%F %T')] [supervisor] $*" | tee -a "$SUP_LOG"; }

# Any trainer process alive in the container? [g]enerative self-match guard
# avoids pgrep matching its own command line.
trainer_alive() {
    local n
    n=$($DOCKER exec "$CONTAINER" bash -lc 'set -f; pgrep -f "[g]enerative_recommenders" | wc -l' 2>/dev/null | tr -dc '0-9')
    [[ "${n:-0}" -gt 0 ]]
}

# Hard-kill any trainer processes left in the container AND wait for GPU HBM to
# actually drain before returning. A rank stuck in a HIP/RCCL collective sits in
# uninterruptible D-state and keeps its multi-hundred-GB embedding shard resident
# for many seconds after SIGKILL; relaunching before that frees makes the next
# attempt OOM on dirty GPUs (an OOM-crash -> dirty-GPU -> OOM cascade). So kill,
# then poll rocm-smi until every GPU is <5 GB (or give up after ~120s).
cleanup_container() {
    $DOCKER exec "$CONTAINER" bash -lc \
        'pkill -9 -f generative_recommenders 2>/dev/null; pkill -9 -f spawn_main 2>/dev/null; pkill -9 -f resource_tracker 2>/dev/null; true' \
        2>/dev/null || true
    local k busy
    for k in $(seq 1 24); do                       # up to ~120s
        busy=$($DOCKER exec "$CONTAINER" bash -lc \
            "rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{if (\$NF+0 > 5e9) c++} END{print c+0}'" \
            2>/dev/null | tr -dc '0-9')
        busy=${busy:-0}
        [[ "$busy" == "0" ]] && return 0
        sup "waiting for GPU HBM to drain ($busy GPU(s) still >5GB)…"
        $DOCKER exec "$CONTAINER" bash -lc 'pkill -9 -f spawn_main 2>/dev/null; true' 2>/dev/null || true
        sleep 5
    done
    sup "WARNING: GPUs still show residual HBM after 120s — launching anyway."
    return 0
}

disk_free_gib() { df -BG --output=avail "$CKPT_PATH" 2>/dev/null | tail -1 | tr -dc '0-9'; }

disk_guard() {
    [[ -z "$CKPT_PATH" ]] && return 0
    local free; free=$(disk_free_gib); free=${free:-0}
    sup "disk guard: ${free} GiB free on $CKPT_PATH (min ${MIN_FREE_GIB})"
    if (( free < MIN_FREE_GIB )); then
        sup "FATAL: insufficient free space (${free} < ${MIN_FREE_GIB} GiB). Aborting."
        return 1
    fi
    return 0
}

# Run the submit-script in the FOREGROUND (its exit status == the trainer's).
# APPEND_LOG=1 preserves the metrics log across relaunches. This is invoked as
# `launch & PID=$!` from the main loop so the backgrounded copy is a DIRECT child
# of this shell — otherwise `wait $PID` can't reap it and always returns 127,
# making every clean completion look like a failure (infinite relaunch loop).
launch() {
    APPEND_LOG=1 CONTAINER="$CONTAINER" DOCKER="$DOCKER" \
        RUN_NAME="$RUN_NAME" LOG="$LOG" CKPT_PATH="$CKPT_PATH" \
        bash "$SUBMIT_SCRIPT" >>"$SUP_LOG" 2>&1
}

sup "=== streaming e2e LOCAL supervisor start ==="
sup "run=$RUN_NAME submit=$SUBMIT_SCRIPT log=$LOG ckpt=$CKPT_PATH container=$CONTAINER"
sup "max_relaunch=$MAX_RELAUNCH min_free_gib=$MIN_FREE_GIB stall_s=$STALL_S poll_s=$POLL_S"

attempt=1
while (( attempt <= MAX_RELAUNCH )); do
    if ! disk_guard; then exit 3; fi
    sup "launching attempt $attempt/$MAX_RELAUNCH"
    cleanup_container                    # ensure no stragglers from a prior attempt
    launch & PID=$!                      # direct child => wait $PID reaps the real rc
    sup "submit-script running as host pid=$PID"

    # --- monitor loop ---
    last_size=0; stall_accum=0; hb=0; hung=0
    while kill -0 "$PID" 2>/dev/null; do
        cur_size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
        if [[ "$cur_size" == "$last_size" ]]; then
            hb=$((hb + 1))
            # Re-check liveness only every 4 polls (cheap docker exec amortized).
            if (( hb % 4 == 0 )); then
                if trainer_alive; then
                    stall_accum=0
                else
                    stall_accum=$((stall_accum + POLL_S * 4))
                    sup "log frozen + no trainer alive (${stall_accum}s/${STALL_S}s)"
                    if (( stall_accum >= STALL_S )); then
                        sup "STALL: hung run — killing pid=$PID + container trainer procs, will relaunch."
                        hung=1
                        kill -9 "$PID" 2>/dev/null || true
                        cleanup_container
                        break
                    fi
                fi
            fi
        else
            stall_accum=0; last_size=$cur_size
        fi
        sleep "$POLL_S"
    done

    # --- the submit-script has exited (or we killed it): decide ---
    wait "$PID" 2>/dev/null; rc=$?
    if (( hung )); then
        sup "attempt $attempt ended via STALL recovery (rc=$rc) — relaunching from latest checkpoint."
    elif (( rc == 0 )); then
        sup "RUN COMPLETED CLEANLY on attempt $attempt."
        sup "=== supervisor done (success) ==="
        exit 0
    else
        sup "attempt $attempt exited rc=$rc (crash/OOM/die_at_step) — relaunching from latest checkpoint."
    fi

    if (( attempt >= MAX_RELAUNCH )); then break; fi
    sleep $(( attempt < 5 ? 20 : 60 ))   # small backoff
    attempt=$((attempt + 1))
done

sup "FATAL: exhausted MAX_RELAUNCH=$MAX_RELAUNCH without completion."
sup "=== supervisor done (failure) ==="
exit 1
