#!/bin/bash
# =============================================================================
# run_streaming_e2e.sh — self-healing supervisor for a yambda-5b streaming
#   train+eval run (sbatch-job level). Works for 1..N nodes.
# =============================================================================
#
# WHAT IT SUPERVISES
#   The run is an `sbatch [--nodes=N] scripts/launch_slurm.sh` BATCH job. That
#   batch script is fully self-contained: it runs orchestrate -> provision
#   (container + RDMA) -> worker (in-container trainer) on EVERY node, so it
#   handles single-node (--nodes=1) and multi-node (world_size=8N) identically.
#   This supervisor wraps THAT job: it monitors it and, on crash / node-failure
#   / hang, RESUBMITS it (which resumes from the latest checkpoint via load
#   auto-latest), bounded by --max-relaunch. There is no docker-exec lifecycle
#   or in-place node failover here — node replacement is SLURM's job on resubmit.
#
# RESUME MODEL (why a resubmit "just works")
#   The trainer checkpoints to $CKPT_PATH and on startup load_dmp_checkpoint
#   auto-resolves to the highest-numbered subdir, restoring model+optimizer+RNG
#   and skipping already-trained batches of a partial window. So resubmitting the
#   SAME submit-script (same CKPT_PATH/LOG) continues from where it died.
#   Resubmits set APPEND_LOG=1 so the metrics log is preserved across attempts.
#
# WHAT IT DETECTS (poll every --poll-s)
#   * job left the queue  -> read sacct State/ExitCode:
#       COMPLETED+0   => run finished (success, exit 0)
#       CANCELLED     => user intent (stop, exit 0 — NOT our place to resubmit)
#       FAILED/NODE_FAIL/TIMEOUT/OUT_OF_MEMORY/BOOT_FAIL/PREEMPTED => relaunch
#   * hang watchdog: job RUNNING but LOG frozen >= --stall-s AND no trainer
#       process alive on ANY node (cross-node pgrep) => scancel + relaunch.
#   * disk guard before each (re)submit: require --min-free-gib on the ckpt vol.
#
# WHERE IT RUNS
#   On the SLURM head node (NFS-mounted /home/chcai code + /apps/chcai
#   ckpts/logs are visible here for squeue/sacct/df and the cross-node pgrep).
#
# USAGE
#   # Submit a fresh job from the launch script, then supervise it:
#   nohup bash scripts/run_streaming_e2e.sh \
#       --submit-script /apps/chcai/yambda_5b_e2e/<run>/launch_1node.sh \
#       --log         /apps/chcai/yambda_5b_e2e/<run>/<run>.log \
#       --ckpt-path   /apps/chcai/yambda_5b_e2e/<run>/ckpts \
#       --run-name    <run> \
#       > /apps/chcai/yambda_5b_e2e/<run>/<run>.supervisor.console.log 2>&1 &
#
#   # Adopt an already-submitted job instead of submitting a new one:
#   nohup bash scripts/run_streaming_e2e.sh --jobid 13235 \
#       --submit-script .../launch_2node.sh --log .../run.log \
#       --ckpt-path .../ckpts --run-name <run> > .../console.log 2>&1 &
#
#   The node count, partition, and reservation all live in the --submit-script's
#   sbatch line (launch_1node.sh / launch_2node.sh / ...), not here.
#
# EXIT CODES
#   0  run completed (COMPLETED+0) or user-cancelled
#   1  exhausted --max-relaunch without completion (or submit failed)
#   3  disk guard tripped
# =============================================================================
set -uo pipefail

JOBID=""                       # adopt this job; empty => submit fresh
SUBMIT_SCRIPT=""
LOG=""
CKPT_PATH=""
RUN_NAME="yambda_5b_e2e"
CONTAINER=yambda_primus
MAX_RELAUNCH=50
MIN_FREE_GIB=1200
STALL_S=2400                   # 40 min: comfortably exceeds a full-holdout eval
                               # window + a blocking ckpt save; only trips when
                               # the log is frozen AND no trainer proc is alive.
POLL_S=30

while [[ $# -gt 0 ]]; do
    case $1 in
        --jobid) JOBID="$2"; shift 2;;
        --submit-script) SUBMIT_SCRIPT="$2"; shift 2;;
        --log) LOG="$2"; shift 2;;
        --ckpt-path) CKPT_PATH="$2"; shift 2;;
        --run-name) RUN_NAME="$2"; shift 2;;
        --container) CONTAINER="$2"; shift 2;;
        --max-relaunch) MAX_RELAUNCH="$2"; shift 2;;
        --min-free-gib) MIN_FREE_GIB="$2"; shift 2;;
        --stall-s) STALL_S="$2"; shift 2;;
        --poll-s) POLL_S="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

[[ -n "$SUBMIT_SCRIPT" && -f "$SUBMIT_SCRIPT" ]] || { echo "FATAL: --submit-script required and must exist"; exit 1; }
[[ -n "$LOG" ]] || { echo "FATAL: --log required"; exit 1; }

SUP_LOG="${LOG%.log}.supervisor.log"
sup() { echo "[$(date '+%F %T')] [supervisor] $*" | tee -a "$SUP_LOG"; }

# Is the job in the queue right now (single read)?
job_in_queue() { [[ -n "$(squeue -h -j "$1" -o '%T' 2>/dev/null | head -1)" ]]; }
job_state()  { squeue -h -j "$1" -o '%T' 2>/dev/null | head -1; }

# Is the job still active? squeue/the SLURM control plane can transiently return
# empty during an NFS/controller blip even though the job is alive (this once
# killed all supervisors at once: empty squeue -> sacct said RUNNING -> a bogus
# "relaunch"). So a SINGLE empty read is not trusted: re-check a few times before
# believing the job is really gone.
job_active() {
    job_in_queue "$1" && return 0
    local k
    for k in 1 2 3; do
        sleep 10
        job_in_queue "$1" && return 0
    done
    return 1
}

# Terminal State + ExitCode from accounting once the job has left the queue.
job_final() { sacct -j "$1" -X -n -o State,ExitCode 2>/dev/null | head -1 | tr -s ' '; }

# sacct/SLURM states that mean the job is STILL ALIVE (not terminal). If we see
# one of these after the monitor loop exits, squeue lied (transient) — resume
# monitoring instead of relaunching (which could spawn a DUPLICATE job).
is_active_state() {
    case "$1" in
        RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|REQUEUE_HOLD|REQUEUE_FED|SIGNALING|STAGE_OUT) return 0;;
        *) return 1;;
    esac
}

# Any trainer process alive on ANY node of the allocation? (cross-node pgrep via
# overlap srun into each node's container). [g]enerative self-match guard avoids
# pgrep matching its own command line.
trainer_alive() {
    local jid="$1" n
    n=$(timeout 70 srun --jobid="$jid" --overlap --ntasks-per-node=1 bash -c \
        "docker exec $CONTAINER bash -lc 'set -f; pgrep -f [g]enerative_recommenders | wc -l' 2>/dev/null" 2>/dev/null \
        | awk '{s+=$1} END{print s+0}')
    [[ "${n:-0}" -gt 0 ]]
}

# Free GiB on the ckpt volume (NFS is mounted on this head node, so df locally).
disk_free_gib() {
    df -BG --output=avail "$CKPT_PATH" 2>/dev/null | tail -1 | tr -dc '0-9'
}

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

# Resubmit the run; resumes from latest checkpoint. APPEND_LOG=1 preserves the
# metrics log. Echoes the new jobid.
resubmit() {
    local out newjid
    out=$(APPEND_LOG=1 bash "$SUBMIT_SCRIPT" 2>&1)
    newjid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | head -1)
    echo "$out" | sed 's/^/    /' >> "$SUP_LOG"
    echo "$newjid"
}

# Submit with retries+backoff. A transient NFS / control-plane error (e.g.
# "sbatch: error: ... I/O error writing script/environment to file") must NOT
# kill the supervisor — it leaves runs unsupervised / unlaunched. Echoes a jobid
# on success, or empty after all retries.
submit_retry() {
    local cand sub_try
    for sub_try in $(seq 1 12); do
        cand=$(resubmit)
        if [[ "$cand" =~ ^[0-9]+$ ]]; then echo "$cand"; return 0; fi
        sup "submit attempt $sub_try/12 failed (transient sbatch/NFS error) — backing off."
        sleep $(( sub_try < 5 ? 30 : 120 ))
    done
    return 1
}

sup "=== streaming e2e supervisor start ==="
sup "run=$RUN_NAME submit=$SUBMIT_SCRIPT log=$LOG ckpt=$CKPT_PATH"
sup "max_relaunch=$MAX_RELAUNCH min_free_gib=$MIN_FREE_GIB stall_s=$STALL_S poll_s=$POLL_S"

attempt=0
if [[ -z "$JOBID" ]]; then
    if ! disk_guard; then exit 3; fi
    sup "no --jobid given; submitting a fresh job"
    JOBID=$(submit_retry)
    [[ "$JOBID" =~ ^[0-9]+$ ]] || { sup "FATAL: could not submit after 12 retries — aborting."; exit 1; }
fi
attempt=1
sup "supervising jobid=$JOBID (attempt $attempt/$MAX_RELAUNCH)"

while (( attempt <= MAX_RELAUNCH )); do
    # --- wait for the job to be schedulable / running ---
    wait_pend=0
    while job_active "$JOBID" && [[ "$(job_state "$JOBID")" != "RUNNING" ]]; do
        (( wait_pend % 10 == 0 )) && sup "job $JOBID state=$(job_state "$JOBID") — waiting to run…"
        sleep "$POLL_S"; wait_pend=$((wait_pend + 1))
    done
    [[ "$(job_state "$JOBID")" == "RUNNING" ]] && sup "job $JOBID RUNNING on $(squeue -h -j "$JOBID" -o '%N' 2>/dev/null | head -1)"

    # --- monitor loop ---
    last_size=0; stall_accum=0; hb=0; self_cancelled=0
    while job_active "$JOBID"; do
        st=$(job_state "$JOBID")
        if [[ "$st" == "RUNNING" ]]; then
            cur_size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
            if [[ "$cur_size" == "$last_size" ]]; then
                # frozen log: only count as a stall if no trainer proc is alive
                # (a long eval / blocking save keeps the process up -> not a stall)
                hb=$((hb + 1))
                if (( hb % 4 == 0 )); then
                    if trainer_alive "$JOBID"; then
                        stall_accum=0
                    else
                        stall_accum=$((stall_accum + POLL_S * 4))
                        sup "log frozen + no trainer alive (${stall_accum}s/${STALL_S}s)"
                        if (( stall_accum >= STALL_S )); then
                            sup "STALL: hung run — scancel $JOBID and relaunch."
                            self_cancelled=1
                            scancel "$JOBID" 2>/dev/null || true
                            sleep 20
                            break
                        fi
                    fi
                fi
            else
                stall_accum=0; last_size=$cur_size
            fi
        fi
        sleep "$POLL_S"
    done

    # --- job has left the queue (or we scancel'd it): decide ---
    sleep 5
    final=$(job_final "$JOBID")
    state=$(echo "$final" | awk '{print $1}')
    code=$(echo "$final" | awk '{print $2}')
    sup "job $JOBID ended: state='${state:-?}' exit='${code:-?}'"

    # The monitor loop only exits when squeue has been empty across several
    # confirming reads. If accounting STILL reports an active state, the job is
    # actually alive (squeue/control-plane blip) — resume monitoring rather than
    # relaunching, which would create a duplicate job.
    if is_active_state "$state"; then
        sup "sacct reports still-active state '$state' — transient squeue blip; resuming monitoring (NOT relaunching)."
        sleep "$POLL_S"
        continue
    fi

    case "$state" in
        COMPLETED)
            if [[ "$code" == "0:0" ]]; then
                sup "RUN COMPLETED CLEANLY on attempt $attempt."
                sup "=== supervisor done (success) ==="
                exit 0
            fi
            sup "COMPLETED but nonzero exit ($code) — relaunching."
            ;;
        CANCELLED*)
            if (( self_cancelled )); then
                sup "job CANCELLED by our own stall recovery — relaunching from latest checkpoint."
            else
                sup "job CANCELLED (user/admin intent) — NOT resubmitting. Stopping supervisor."
                sup "=== supervisor done (cancelled) ==="
                exit 0
            fi
            ;;
        FAILED|NODE_FAIL|TIMEOUT|OUT_OF_MEMORY|BOOT_FAIL|PREEMPTED|"")
            sup "failure state '${state:-unknown}' — will relaunch from latest checkpoint."
            ;;
        *)
            sup "unrecognized terminal state '${state}' — relaunching to be safe."
            ;;
    esac

    if (( attempt >= MAX_RELAUNCH )); then break; fi
    if ! disk_guard; then exit 3; fi
    sleep $(( attempt < 5 ? 20 : 60 ))   # small backoff
    # Resubmit with retries. A transient NFS / control-plane error (e.g.
    # "sbatch: error: Batch job submission failed: I/O error writing
    # script/environment to file") must NOT kill the supervisor — that once
    # left a live run permanently unsupervised. Retry with backoff first.
    JOBID=$(submit_retry)
    if ! [[ "$JOBID" =~ ^[0-9]+$ ]]; then
        sup "FATAL: resubmit failed after 12 retries — aborting."; exit 1
    fi
    attempt=$((attempt + 1))
    sup "relaunched as jobid=$JOBID (attempt $attempt/$MAX_RELAUNCH)"
done

sup "FATAL: exhausted MAX_RELAUNCH=$MAX_RELAUNCH without completion."
sup "=== supervisor done (failure) ==="
exit 1
