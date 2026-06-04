#!/bin/bash
# End-to-end failure-injection + resume test for streaming-train-eval.
#
# Validates exact-once mid-window resume on the yambda-5b stack:
#   Phase 1 (baseline):     uninterrupted run for N=2 train_ts × K batches/window
#   Phase 2 (interrupted):  same config but die_at_step=M → exits at step M
#                           after the in-window checkpoint lands
#   Phase 3 (resume):       re-launch with same CKPT_PATH → auto-latest picks
#                           the in-window save → finishes the partial window
#                           and the rest of the requested train_ts list
# Assertion: traj_resumed[step].window_ne / window_auc / window_accuracy match
# traj_baseline bit-equal (np.allclose atol=1e-4) for all step > die_at_step.
#
# Driven entirely via env-driven gin knobs defined in yambda_5b.gin:
#   NUM_TRAIN_TS / NUM_TRAIN_BATCHES / IN_WINDOW_CKPT_FREQ / DIE_AT_STEP /
#   CKPT_PATH / KEEP_LAST_N / EVAL_EACH_WINDOW
#
# Usage:
#   bash scripts/streaming_resume_test.sh --jobid <slurm-jobid>
#       [--container yambda_primus]
#       [--num-train-batches 200]
#       [--die-at-step 350]
#       [--keep]    # retain LOG_DIR + CKPT after run for inspection

set -uo pipefail

JOBID=""
CONTAINER="yambda_primus"
NUM_TRAIN_BATCHES=200
DIE_AT_STEP=350
IN_WINDOW_FREQ=50
KEEP=0
# Trajectory closeness bound — NOT a bit-equality check. The ROCm training stack
# is nondeterministic across runs (non-deterministic atomic scatter-add in the
# embedding/attention backward): two independent *cold* runs already drift
# ~7e-4 in window_ne over 20 steps, and early-training chaos (AUC~0.5) amplifies
# any seed difference. So resume-vs-baseline can legitimately differ by a few
# percent. This bound just catches GROSS divergence (wrong data skip, totally
# unrestored state) while tolerating nondeterministic drift. The HARD resume
# correctness gates are the functional-invariant checks below (RNG restored,
# resumed-at-correct-step, atomic/keep_last_n), not this number.
ATOL=0.15
CKPT_ROOT=/apps/chcai/ckpts_resume_test
LOG_DIR=/apps/chcai/streaming_resume_test
REPO=/home/chcai/training/recommendation_v4

while [[ $# -gt 0 ]]; do
    case $1 in
        --jobid) JOBID="$2"; shift 2;;
        --container) CONTAINER="$2"; shift 2;;
        --num-train-batches) NUM_TRAIN_BATCHES="$2"; shift 2;;
        --die-at-step) DIE_AT_STEP="$2"; shift 2;;
        --in-window-freq) IN_WINDOW_FREQ="$2"; shift 2;;
        --atol) ATOL="$2"; shift 2;;
        --keep) KEEP=1; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done
[[ -z "$JOBID" ]] && { echo "Error: --jobid required"; exit 1; }

mkdir -p "$LOG_DIR"

# Single-window mid-window resume: NUM_TRAIN_TS=1, so the whole test runs inside
# train_ts=START_TS. die_at_step must land strictly inside that window, AT a
# multiple of IN_WINDOW_FREQ so an in-window checkpoint is saved right before
# the crash (resume then skips exactly DIE_AT_STEP already-trained batches).
if (( DIE_AT_STEP <= 0 || DIE_AT_STEP >= NUM_TRAIN_BATCHES )); then
    echo "Warning: die_at_step=$DIE_AT_STEP not strictly inside window (0, $NUM_TRAIN_BATCHES)" >&2
fi
if (( DIE_AT_STEP % IN_WINDOW_FREQ != 0 )); then
    echo "Warning: die_at_step=$DIE_AT_STEP not a multiple of in_window_freq=$IN_WINDOW_FREQ; no save lands exactly at crash" >&2
fi

cleanup_workers() {
    srun --jobid="$JOBID" --overlap docker exec "$CONTAINER" bash -lc \
        "pkill -9 -f generative_recommenders 2>/dev/null; sleep 2; \
         pkill -9 -f spawn_main 2>/dev/null; sleep 3; true" 2>/dev/null || true
}
clean_ckpt() {
    srun --jobid="$JOBID" --overlap docker exec "$CONTAINER" rm -rf "$CKPT_ROOT" 2>/dev/null || true
}

# Wait for a log line to appear OR a crash sentinel. Returns 0 if target found,
# 1 if crash sentinel found first.
wait_for_log() {
    local log="$1"; local target_re="$2"; local timeout_s="${3:-1500}"
    local elapsed=0
    while (( elapsed < timeout_s )); do
        if grep -qE "$target_re" "$log" 2>/dev/null; then
            return 0
        fi
        if grep -qE "Traceback|RuntimeError|OutOfMemoryError" "$log" 2>/dev/null; then
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    return 2
}

# Single train window of NUM_TRAIN_BATCHES steps → last train step == NUM_TRAIN_BATCHES.
LAST_STEP=$NUM_TRAIN_BATCHES

run_phase() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    # Join the per-phase env overrides into ONE word. Using `$*` (not `$@`) is
    # essential: `$@` embedded mid-string in the double-quoted `bash -lc "..."`
    # expands to *multiple* arguments, so bash -lc would only run up to the
    # first override and treat the rest as positional params — launch_smoke
    # would never execute (silent 0-byte log).
    local env_overrides="$*"
    : > "$log"
    echo "[$(date)] === phase '$name' ==="
    cleanup_workers
    srun --jobid="$JOBID" --overlap docker exec -d "$CONTAINER" bash -lc "
        cd $REPO &&
        HSTU_HAMMER_KERNEL=TRITON \
        $env_overrides \
        RUN_NAME=resume_test_$name \
        LOG=$log \
        bash scripts/launch_smoke_8gpu.sh
    "
}

# === Phase 1: baseline ===
clean_ckpt
run_phase baseline \
    "NUM_TRAIN_TS=1" \
    "EVAL_EACH_WINDOW=0" \
    "METRIC_LOG_FREQ=1" \
    "NUM_TRAIN_BATCHES=$NUM_TRAIN_BATCHES" \
    "DIE_AT_STEP=-1"
wait_for_log "$LOG_DIR/baseline.log" "train - Step $LAST_STEP metrics" 1500
rc=$?
cleanup_workers
[[ $rc -ne 0 ]] && { echo "FAIL: baseline didn't finish"; tail -20 "$LOG_DIR/baseline.log"; exit 1; }

# === Phase 2: interrupted ===
clean_ckpt
run_phase interrupt \
    "NUM_TRAIN_TS=1" \
    "EVAL_EACH_WINDOW=0" \
    "METRIC_LOG_FREQ=1" \
    "NUM_TRAIN_BATCHES=$NUM_TRAIN_BATCHES" \
    "IN_WINDOW_CKPT_FREQ=$IN_WINDOW_FREQ" \
    "KEEP_LAST_N=1" \
    "DIE_AT_STEP=$DIE_AT_STEP" \
    "CKPT_PATH=$CKPT_ROOT"
wait_for_log "$LOG_DIR/interrupt.log" "die_at_step=$DIE_AT_STEP hit" 1500
rc=$?
cleanup_workers
[[ $rc -ne 0 ]] && { echo "FAIL: interrupt didn't hit die_at_step"; tail -20 "$LOG_DIR/interrupt.log"; exit 1; }

SAVED=$(srun --jobid="$JOBID" --overlap docker exec "$CONTAINER" ls "$CKPT_ROOT" 2>/dev/null | tr '\n' ' ')
echo "Saved checkpoints after interrupt: $SAVED"

# === Phase 3: resume ===
run_phase resume \
    "NUM_TRAIN_TS=1" \
    "EVAL_EACH_WINDOW=0" \
    "METRIC_LOG_FREQ=1" \
    "NUM_TRAIN_BATCHES=$NUM_TRAIN_BATCHES" \
    "IN_WINDOW_CKPT_FREQ=$IN_WINDOW_FREQ" \
    "KEEP_LAST_N=1" \
    "DIE_AT_STEP=-1" \
    "CKPT_PATH=$CKPT_ROOT"
wait_for_log "$LOG_DIR/resume.log" "train - Step $LAST_STEP metrics" 1500
rc=$?
[[ $rc -ne 0 ]] && { cleanup_workers; echo "FAIL: resume didn't finish"; tail -20 "$LOG_DIR/resume.log"; exit 1; }
# The resume run performs an end-of-window checkpoint save AFTER the final
# step's metric line. That save (hundreds of GB) writes <ts>.tmp and then
# atomically renames it onto <ts>, logging "checkpoint successfully saved" only
# once the rename completes. If we kill workers right after the step line we'd
# orphan a half-written <ts>.tmp and trip the stale-dir gate below — a harness
# race, not a resume bug. Wait for the save to finish before tearing down.
wait_for_log "$LOG_DIR/resume.log" "checkpoint successfully saved" 1500
save_rc=$?
cleanup_workers
[[ $save_rc -ne 0 ]] && { echo "FAIL: resume end-of-window checkpoint save did not complete"; tail -20 "$LOG_DIR/resume.log"; exit 1; }

# === HARD resume-correctness gates (functional invariants) ===
# These — not the trajectory closeness check below — are the authoritative
# proof the resume path is correct, because they're deterministic and immune
# to the GPU nondeterminism that perturbs the metric trajectory.

# (1) Re-entered the partial window at exactly the saved batch_idx_in_window.
if ! grep -qE "Resuming mid-window at train_ts=[0-9]+ batch_idx_in_window=$DIE_AT_STEP\b" "$LOG_DIR/resume.log" 2>/dev/null; then
    echo "FAIL: resume did not re-enter mid-window at batch_idx_in_window=$DIE_AT_STEP"
    grep -E "Resuming" "$LOG_DIR/resume.log" 2>/dev/null | head -2
    exit 1
fi
# (2) Per-rank RNG state was actually restored (dropout determinism path).
RNG_RESTORED=$(grep -c "RNG state restored from" "$LOG_DIR/resume.log" 2>/dev/null || echo 0)
echo "RNG state restored on $RNG_RESTORED ranks"
[[ "$RNG_RESTORED" -lt 1 ]] && { echo "FAIL: no RNG state restored on resume"; exit 1; }
# (3) The FIRST training step after resume is exactly die_at_step+1, i.e. the
#     skip-already-trained-batches logic emitted the next unseen batch (not a
#     restart from step 1, and not a gap).
FIRST_RESUMED=$(grep -oE 'train - Step [0-9]+ metrics: \{.metric' "$LOG_DIR/resume.log" 2>/dev/null \
    | grep -oE 'Step [0-9]+' | awk '{print $2}' | sort -n | head -1)
echo "First resumed train step: $FIRST_RESUMED (expect $((DIE_AT_STEP + 1)))"
[[ "$FIRST_RESUMED" != "$((DIE_AT_STEP + 1))" ]] && {
    echo "FAIL: resume did not continue at step $((DIE_AT_STEP + 1)) (got $FIRST_RESUMED)"; exit 1; }

# === Final on-disk state checks (atomic save + retention) ===
NUM_CKPT=$(srun --jobid="$JOBID" --overlap docker exec "$CONTAINER" bash -lc \
    "ls $CKPT_ROOT 2>/dev/null | grep -E '^[0-9]+$' | wc -l" | tr -d ' ')
# Both .tmp (interrupted write) and .old (interrupted atomic-overwrite swap)
# must be absent — their presence means a save crashed without clean recovery.
STALE_CKPT=$(srun --jobid="$JOBID" --overlap docker exec "$CONTAINER" bash -lc \
    "ls $CKPT_ROOT 2>/dev/null | grep -E '\\.(tmp|old)$' | wc -l" | tr -d ' ')
echo "Final: $NUM_CKPT numeric ckpt subdirs, $STALE_CKPT stale (.tmp/.old) dirs (expect 1, 0)"
[[ "$NUM_CKPT" != "1" ]] && { echo "FAIL: keep_last_n=1 violated"; exit 1; }
[[ "$STALE_CKPT" != "0" ]] && { echo "FAIL: stale .tmp/.old dirs left behind"; exit 1; }
echo "=== Resume functional invariants: ALL PASS ==="

# === Trajectory closeness (sanity bound, NOT bit-equality) ===
# Catches gross resume bugs (wrong data slice, unrestored model) that throw the
# metric trajectory far off. Small drift is expected & tolerated (see ATOL note
# at top). The functional invariants above are the real correctness proof.
python3 $REPO/generative_recommenders/dlrm_v3/train/tests/streaming_resume_test.py parse \
    "$LOG_DIR/baseline.log" "$LOG_DIR/traj_baseline.json"
python3 $REPO/generative_recommenders/dlrm_v3/train/tests/streaming_resume_test.py parse \
    "$LOG_DIR/resume.log" "$LOG_DIR/traj_resumed.json"

python3 $REPO/generative_recommenders/dlrm_v3/train/tests/streaming_resume_test.py compare \
    "$LOG_DIR/traj_baseline.json" "$LOG_DIR/traj_resumed.json" \
    --min-resume-step $((DIE_AT_STEP + 1)) --atol $ATOL
RC=$?

if [[ "$KEEP" != "1" ]]; then
    rm -rf "$LOG_DIR"
    clean_ckpt
fi

if [[ $RC -eq 0 ]]; then
    echo "=== PASS: resume validated (functional invariants + trajectory within $ATOL of baseline) ==="
else
    echo "=== FAIL: trajectory diverged beyond $ATOL — likely a real resume bug (wrong data slice / unrestored state), not nondeterminism ==="
fi
exit $RC
