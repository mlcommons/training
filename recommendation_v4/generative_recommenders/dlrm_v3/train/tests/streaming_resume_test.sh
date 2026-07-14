#!/bin/bash
# End-to-end failure-injection + resume test for streaming-train-eval.
#
# ============================================================================
# WHY TWO TESTS (the intuition, in plain language)
# ============================================================================
# Training runs over consecutive time WINDOWS (window 0, then 1, then 2, ...).
# All N GPUs must march from one window to the next IN LOCKSTEP: they constantly
# do "everybody-talk-at-once" group ops (NCCL collectives — sharing embeddings
# across GPUs), and every GPU must enter each group-op at the same time. If one
# GPU is late, the rest wait for it forever and the whole job FREEZES (deadlock).
#
# The two scenarios check DIFFERENT kinds of failure — not bigger vs smaller:
#
#   midwindow   = CORRECTNESS. "When I crash and resume, do I land on the exact
#                 right batch with the right RNG state and produce the same
#                 numbers?"  (stays inside ONE window; never crosses a seam.)
#
#   multiwindow = LIVENESS / NO-DEADLOCK. "Can all N GPUs hand off across a
#                 window SEAM together without one falling out of step and
#                 hanging the job?"  (needs >=2 windows so a seam actually exists.)
#
# The dangerous spot is the SEAM between two windows: there, each GPU does solo
# prep work (load next window's data; count anchors for the eval cadence) and
# they DON'T all finish at the same speed. Two bugs lived exactly there, and BOTH
# are invisible to a single-window test:
#   (A) every GPU separately ran a slow O(N) "count all the data" pass -> they
#       finished at different times -> fast GPU barged into the next group-op
#       while others were still counting -> freeze.
#       FIX: only rank 0 counts, then broadcasts the number to everyone else.
#   (B) no rendezvous at the seam -> uneven data-prep -> same desync -> freeze.
#       FIX: a dist.barrier() at every window boundary (all GPUs wait, then cross
#       together).  WINDOW_BARRIER_DEBUG=1 makes rank 0 log one line per seam.
#
#   TIMELINE — without the fixes (each GPU on its own clock at the seam):
#       win0 train  | solo prep (varies) | next group-op
#       GPU0 ########|=====|              >> waiting..........
#       GPU1 ########|========|           >> waiting.......
#       GPU2 ########|===========|        >> waiting....
#       GPU3 ########|==============|     >> never lines up  -> HANG
#
#   TIMELINE — with the fixes (rank 0 counts + a barrier gate at the seam):
#       win0 train  | [== BARRIER: all wait ==] | win1 train
#       GPU0 ########| count |  wait           # |########
#       GPU1 ########|       |  wait           # |########
#       GPU2 ########|       |  wait           # |########
#       GPU3 ########|       |  wait           # |########
#                 rank0 shares count ^   all cross together ^  -> OK
#
# Why midwindow can NOT catch (A)/(B): it runs a SINGLE window with per-window
# eval off (NUM_TRAIN_TS=1, EVAL_EVERY_N_WINDOWS=0, split=1.0), so it never
# reaches a seam and never turns on the data-fraction-eval/anchor-count path.
# A broken barrier or broken broadcast passes midwindow silently.
#
# Why the multiwindow RESUME phase (P3 below) is the meanest case: restarting
# from a checkpoint loads the saved window and then IMMEDIATELY steps across a
# seam into the next window — landing right on the spot that used to freeze, AND
# re-running all that slow setup on the resume path. If (A)/(B) regressed, P3
# hangs and the test fails by timing out.
#
#                 | midwindow            | multiwindow
#   --------------+----------------------+-----------------------------
#   proves        | resume to RIGHT spot | cross seam WITHOUT freezing
#   windows       | 1 (no seam)          | >=2 (crosses >=1 seam)
#   data-pct eval | off                  | on (exercises the anchor count)
#   catches       | wrong batch/RNG/ckpt | missing barrier/broadcast -> HANG
#   failure mode  | wrong NUMBERS        | job FREEZES forever
# They are complementary: you need BOTH.
# ============================================================================
#
# PLATFORM-GENERAL: runs on both NVIDIA B200 and AMD MI350/MI355 (ROCm/meta64).
# The only hardware-specific bits are picked by --platform (auto-detected from the
# running container if omitted): the container name, the dataset path, and the
# checkpoint root. Everything else — the worker entrypoint (scripts/launch_slurm.sh,
# which is the shared launcher both clusters' supervisors use), the env-driven gin
# knobs, and all assertions — is identical across platforms.
#
# Two scenarios (select with --scenario; default runs both):
#
#  midwindow  — exact-once MID-WINDOW resume (single window).
#     P1 baseline:    uninterrupted 1 train_ts × K batches.
#     P2 interrupted: same + die_at_step=M → exits AFTER the in-window ckpt at M.
#     P3 resume:      relaunch w/ same CKPT_PATH → auto-latest picks the in-window
#                     save, skips the M already-trained batches, finishes.
#     Gates: re-entered at batch_idx_in_window=M, per-rank RNG restored, first
#     resumed step == M+1, atomic save + keep_last_n, trajectory within --atol.
#
#  multiwindow — distributed-sync REGRESSION guard for the two fixes the
#     mid-window test cannot reach (it runs ONE window with per-window eval off):
#        (A) total_train_anchors() computed ONCE on rank 0 + broadcast (the
#            data-fraction eval cadence needs it; running the multi-minute O(N)
#            mmap gather + uid-hash on every rank desynced NCCL → boundary hang).
#        (B) a dist.barrier() at every window boundary before the first forward
#            (per-rank data-prep skew otherwise desyncs the collective stream).
#     Both only bite across >=2 windows with EVAL_EVERY_DATA_PCT>0, and the
#     deadlock struck at a boundary mid-run, so:
#        P1 mw_baseline: cold run over MW_TS windows w/ data-pct eval. Asserts
#            total_train_anchors logged EXACTLY ONCE (computed at setup + broadcast
#            from rank 0), the barrier fired on EVERY window, the data-pct cadence
#            was set up, and the run COMPLETED (no boundary hang).
#        P2 mw_seed:     1 window → clean end-of-window (WINDOW_COMPLETE) ckpt.
#        P3 mw_resume:   relaunch over MW_TS windows w/ same CKPT_PATH → resumes
#            past the completed window and CROSSES the boundary into the next
#            windows. Asserts "Resuming from completed", barrier fired on each
#            remaining window, anchors broadcast once, and the run COMPLETED —
#            i.e. the exact boundary-crossing-on-resume case that used to hang.
#
# Driven entirely via env-driven gin knobs (yambda_5b.gin) through the SAME worker
# entrypoint both platforms' production supervisors use: `bash scripts/launch_slurm.sh`
# (worker phase, auto-detected inside the container). WINDOW_BARRIER_DEBUG=1 makes
# the otherwise-silent barrier emit one rank-0 line per crossed window.
#
# CHECKPOINT/DATASET PLACEMENT (the one real platform difference):
#   * B200: virtiofs/NFS WEDGES under the trainer's concurrent mmap LOAD, so the
#     checkpoint root AND the mmap'd dataset cache MUST be node-local (defaults
#     /tmp/...). The dataset must already be staged node-local at --data-path
#     (the e2e supervisor's stage_data_in does this); the test fails fast if not.
#   * MI350/MI355 (meta64): NFS mmap is fine, so the checkpoint root + dataset
#     read directly from shared NFS (defaults /apps/chcai/...), as the original
#     test did. No staging needed.
#   Logs always use read()/write() only, so they live on shared /apps/chcai and
#   are grep-able from the head node on both platforms.
#
# Usage:
#   bash generative_recommenders/dlrm_v3/train/tests/streaming_resume_test.sh \
#       --jobid <slurm-jobid> [--platform b200|mi350] [--scenario all]
#       [--container <name>] [--data-path <path>] [--ckpt-root <path>] [--start-ts 150]
#       [--num-train-batches 200] [--die-at-step 100]      # midwindow knobs
#       [--mw-num-train-ts 3] [--mw-num-train-batches 20]  # multiwindow knobs
#       [--mw-eval-pct 0.34] [--phase-timeout S] [--mw-run-timeout S] [--keep]
#   --platform is auto-detected from the running container when omitted. Any of
#   --container/--data-path/--ckpt-root override the platform default.
#   Per-phase wait budgets default per-platform (B200 node-local NVMe: 1800/3600s;
#   MI350/MI355 shared-NFS full-model ckpts ~9 min each: 5400/5400s) and can be
#   overridden with --phase-timeout (midwindow) / --mw-run-timeout (multiwindow).

set -uo pipefail

JOBID=""
# Repo root is derived from THIS script's location
# (<repo>/generative_recommenders/dlrm_v3/train/tests/streaming_resume_test.sh —
# four levels up), so the test is not pinned to any one user's home. Override with
# --repo if the repo is mounted at a different path inside the container.
_SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$_SELF_DIR/../../../.." && pwd)
DATASET_SUBDIR=processed_5b/hstu_cache_L4086
SCENARIO=all                                # midwindow | multiwindow | all
START_TS=150
KEEP=0
LOG_DIR=/apps/chcai/streaming_resume_test   # shared NFS (read()/write() only)
# Platform + the three platform-specific paths. Empty sentinels here; filled by
# apply_platform_defaults() AFTER platform detection unless the user overrode
# them on the command line. (DATA_PATH uses a distinct sentinel because an
# explicit empty value is meaningful: "do not inject DLRM_DATA_PATH; let the gin
# default apply".)
PLATFORM=""                                 # b200 | mi350 | mi355 ; auto if empty
CONTAINER=""                                # default: per-platform
DATA_PATH="__AUTO__"                         # default: per-platform
CKPT_ROOT=""                                # default: per-platform (node-local on B200)

# --- midwindow knobs ---
NUM_TRAIN_BATCHES=200
NUM_EVAL_BATCHES=5      # cap the per-phase FINAL eval (0 = full holdout, very slow)
DIE_AT_STEP=100
IN_WINDOW_FREQ=50
ATOL=0.15           # trajectory closeness bound (NOT bit-equality; see py module)
# Per-phase wait budget. Left empty here and filled per-platform below (a B200
# ckpt save/load hits node-local NVMe and is fast; on meta64 each full-model DCP
# save/load lands on shared NFS and takes ~9 min, and the resume phase does a
# LOAD + several in-window saves + an end-of-window save, so it needs far longer).
# Override explicitly with --phase-timeout.
MW_TIMEOUT=""

# --- multiwindow knobs ---
MW_TS=3                 # windows to train (>=2 to cross a boundary)
MW_BATCHES=20           # train batches per window (small = fast)
MW_EVAL_BATCHES=5       # holdout eval batches per fired eval
MW_EVAL_PCT=0.34        # data-fraction eval cadence (>0 enables the anchors path)
MW_SPLIT=0.90           # train split (<1 => holdout exists => uid-hash anchor path)
MW_HOLDOUT_TS=200       # PINNED holdout window (must match across seed→resume)
# generous: init + planner + anchors gather can take min; on NFS add ckpt save/load.
# Empty => filled per-platform below. Override with --mw-run-timeout.
MW_RUN_TIMEOUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --jobid) JOBID="$2"; shift 2;;
        --platform) PLATFORM="$2"; shift 2;;
        --container) CONTAINER="$2"; shift 2;;
        --repo) REPO="$2"; shift 2;;
        --data-path) DATA_PATH="$2"; shift 2;;
        --dataset-subdir) DATASET_SUBDIR="$2"; shift 2;;
        --scenario) SCENARIO="$2"; shift 2;;
        --start-ts) START_TS="$2"; shift 2;;
        --ckpt-root) CKPT_ROOT="$2"; shift 2;;
        --log-dir) LOG_DIR="$2"; shift 2;;
        --num-train-batches) NUM_TRAIN_BATCHES="$2"; shift 2;;
        --num-eval-batches) NUM_EVAL_BATCHES="$2"; shift 2;;
        --die-at-step) DIE_AT_STEP="$2"; shift 2;;
        --in-window-freq) IN_WINDOW_FREQ="$2"; shift 2;;
        --atol) ATOL="$2"; shift 2;;
        --phase-timeout) MW_TIMEOUT="$2"; shift 2;;
        --mw-run-timeout) MW_RUN_TIMEOUT="$2"; shift 2;;
        --mw-num-train-ts) MW_TS="$2"; shift 2;;
        --mw-num-train-batches) MW_BATCHES="$2"; shift 2;;
        --mw-num-eval-batches) MW_EVAL_BATCHES="$2"; shift 2;;
        --mw-eval-pct) MW_EVAL_PCT="$2"; shift 2;;
        --mw-split) MW_SPLIT="$2"; shift 2;;
        --mw-holdout-ts) MW_HOLDOUT_TS="$2"; shift 2;;
        --keep) KEEP=1; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done
[[ -z "$JOBID" ]] && { echo "Error: --jobid required"; exit 1; }
case "$SCENARIO" in midwindow|multiwindow|all) ;; *) echo "Error: --scenario must be midwindow|multiwindow|all"; exit 1;; esac
(( MW_TS < 2 )) && { echo "Error: --mw-num-train-ts must be >=2 to cross a boundary"; exit 1; }
[[ -n "$PLATFORM" ]] && case "$PLATFORM" in b200|mi350|mi355) ;; *) echo "Error: --platform must be b200|mi350|mi355"; exit 1;; esac

# --- resolve platform + its three hardware-specific paths --------------------
# Precedence: explicit --platform > inferred from explicit --container > probe
# the allocation's docker for a known training container > default b200.
if [[ -z "$PLATFORM" ]]; then
    if [[ "$CONTAINER" == "yambda_b200" ]]; then PLATFORM=b200
    elif [[ "$CONTAINER" == "yambda_primus" ]]; then PLATFORM=mi350
    else
        _names=$(srun --jobid="$JOBID" --overlap docker ps -a --format '{{.Names}}' 2>/dev/null)
        if grep -qx yambda_b200 <<<"$_names"; then PLATFORM=b200
        elif grep -qx yambda_primus <<<"$_names"; then PLATFORM=mi350
        else
            # No known training container yet (e.g. container not provisioned).
            # Fall back to probing the allocation's GPU vendor on the host so we
            # do NOT silently assume a platform.
            _vendor=$(srun --jobid="$JOBID" --overlap bash -lc \
                'if command -v rocm-smi >/dev/null 2>&1; then echo amd; \
                 elif command -v nvidia-smi >/dev/null 2>&1; then echo nvidia; \
                 else echo unknown; fi' 2>/dev/null | head -1)
            case "$_vendor" in
                amd)    PLATFORM=mi350; echo "[$(date)] no known container — detected AMD GPU host (rocm-smi) → mi350";;
                nvidia) PLATFORM=b200;  echo "[$(date)] no known container — detected NVIDIA GPU host (nvidia-smi) → b200";;
                *) echo "Error: could not auto-detect platform on job $JOBID (no yambda_b200/yambda_primus container and no rocm-smi/nvidia-smi). Pass --platform b200|mi350|mi355."; exit 1;;
            esac
        fi
    fi
    echo "[$(date)] auto-detected platform: $PLATFORM"
fi
case "$PLATFORM" in
    b200)
        : "${CONTAINER:=yambda_b200}"
        # B200: mmap (ckpt LOAD + dataset cache) must NOT touch virtiofs/NFS.
        [[ "$DATA_PATH" == "__AUTO__" ]] && DATA_PATH=/tmp/yambda_data
        : "${CKPT_ROOT:=/tmp/yambda_resume_test/ckpts}"
        # Node-local NVMe: full-model save/load is fast.
        : "${MW_TIMEOUT:=1800}"
        : "${MW_RUN_TIMEOUT:=3600}"
        ;;
    mi350|mi355)
        : "${CONTAINER:=yambda_primus}"
        # meta64: NFS mmap is fine — read dataset + write ckpt directly on NFS
        # (matches the original MI350 test). /apps/chcai/dlrm_data is the gin default.
        [[ "$DATA_PATH" == "__AUTO__" ]] && DATA_PATH=/apps/chcai/dlrm_data
        : "${CKPT_ROOT:=/apps/chcai/ckpts_resume_test}"
        # Shared NFS: each full-model DCP save/load is ~9 min. The midwindow resume
        # phase chains a LOAD + multiple in-window saves + an end-of-window save
        # (>2000s observed), so the B200 budgets are far too tight — abandoning a
        # still-running trainer leaks GPU VRAM and OOMs the next phase. Be generous.
        : "${MW_TIMEOUT:=5400}"
        : "${MW_RUN_TIMEOUT:=5400}"
        ;;
esac
echo "[$(date)] platform=$PLATFORM container=$CONTAINER data_path=${DATA_PATH:-<gin default>} ckpt_root=$CKPT_ROOT phase_timeout=${MW_TIMEOUT}s mw_run_timeout=${MW_RUN_TIMEOUT}s"

mkdir -p "$LOG_DIR"
PYHELPER="$REPO/generative_recommenders/dlrm_v3/train/tests/streaming_resume_test.py"

# --- container helpers (inspect CKPT/dataset via docker exec — works whether the
#     path is node-local on B200 or shared NFS on MI350) ---
sx() { srun --jobid="$JOBID" --overlap docker exec "$CONTAINER" bash -lc "$1" 2>/dev/null; }

# Kill any lingering trainer procs from a prior phase AND block until they are
# really gone, so the freed GPU VRAM is reclaimed before the next phase shards
# its embedding tables (otherwise it OOMs on the leaked memory).
#   * Bracketed patterns ([t]rain_ranker, …) are REQUIRED: a plain `pkill -f
#     train_ranker` issued inside `bash -lc "...train_ranker..."` matches its OWN
#     command line and SIGKILLs this very shell (docker exec returns 137), which
#     silently aborted the rest of the old cleanup and leaked trainers/VRAM.
#   * After signalling, poll until no trainer remains (bounded), then a short
#     settle so the driver finishes reclaiming device memory.
cleanup_workers() {
    sx '
        for pat in "[t]rain_ranker" "[g]enerative_recommenders" "[s]pawn_main" "[m]ultiprocessing"; do
            pkill -9 -f "$pat" 2>/dev/null
        done
        for _ in $(seq 1 30); do
            pgrep -f "[t]rain_ranker" >/dev/null 2>&1 || \
            pgrep -f "[g]enerative_recommenders" >/dev/null 2>&1 || break
            sleep 2
        done
        sleep 3; true' || true
}
clean_ckpt() { sx "rm -rf '$1'" || true; }

# Precheck: the dataset cache must be readable at $DATA_PATH. On B200 it must be
# staged node-local (the supervisor's stage_data_in does this) since mmap from
# virtiofs/NFS wedges; on MI350 it reads directly from NFS. Skipped when DATA_PATH
# is empty (the trainer falls back to its gin default and we don't know the path).
precheck_data() {
    [[ -z "$DATA_PATH" ]] && { echo "[$(date)] data path unset — trainer will use its gin default; skipping precheck"; return 0; }
    local ok
    ok=$(sx "[ -d '$DATA_PATH/$DATASET_SUBDIR' ] && echo yes || echo no")
    if [[ "$ok" != "yes" ]]; then
        echo "FAIL: dataset cache not found at $DATA_PATH/$DATASET_SUBDIR inside '$CONTAINER' (platform=$PLATFORM)."
        if [[ "$PLATFORM" == "b200" ]]; then
            echo "      B200: stage it node-local first (the e2e supervisor does this via stage_data_in),"
            echo "      or pass --data-path to an already-staged local mirror. mmap from virtiofs/NFS wedges."
        else
            echo "      MI350/MI355: pass --data-path to the NFS dataset root (gin default is /apps/chcai/dlrm_data)."
        fi
        exit 1
    fi
}

# Wait (host-side grep on the shared-NFS log) for a target regex OR a crash
# sentinel. 0=target found, 1=crash first, 2=timeout.
wait_for_log() {
    local log="$LOG_DIR/$1.log"; local target_re="$2"; local timeout_s="${3:-1800}"
    local elapsed=0
    while (( elapsed < timeout_s )); do
        grep -qE "$target_re" "$log" 2>/dev/null && return 0
        grep -qE "Traceback|RuntimeError|OutOfMemoryError|CUDA error" "$log" 2>/dev/null && return 1
        sleep 5; elapsed=$((elapsed + 5))
    done
    return 2
}

# Launch one trainer phase (detached), appending a PHASE_EXIT sentinel after the
# trainer returns (clean OR crash) — exactly like the production supervisor. The
# common env (data path, mode, start_ts, barrier debug) is fixed; per-phase knobs
# are passed as additional "K=V" words.
run_phase() {
    local name="$1"; shift
    local log="$LOG_DIR/${name}.log"
    # `$*` (joined into ONE word), NOT `$@`: embedded mid-string in the
    # double-quoted `bash -lc "..."`, `$@` would expand to multiple args and
    # bash -lc would stop after the first override (silent 0-byte log).
    local env_overrides="$*"
    # Inject DLRM_DATA_PATH only when a path is set; an empty DATA_PATH means
    # "use the trainer's gin default" (the meta64 NFS root).
    local data_env=""
    [[ -n "$DATA_PATH" ]] && data_env="DLRM_DATA_PATH=$DATA_PATH"
    : > "$log"
    echo "[$(date)] === phase '$name' ==="
    cleanup_workers
    srun --jobid="$JOBID" --overlap docker exec -d "$CONTAINER" bash -lc "
        cd $REPO &&
        $data_env \
        HSTU_HAMMER_KERNEL=TRITON \
        MODE=streaming-train-eval \
        START_TS=$START_TS \
        WINDOW_BARRIER_DEBUG=1 \
        RUN_NAME=resume_test_$name \
        LOG=$log \
        $env_overrides \
        bash scripts/launch_slurm.sh;
        echo \"PHASE_EXIT=\$? \$(date '+%F %T')\" >> $log
    "
}

# Read a scalar field from a summarize-JSON.
jget() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]))" "$1" "$2"; }

FAIL=0
fail() { echo "FAIL: $*"; FAIL=1; }

precheck_data

# =============================================================================
# SCENARIO: midwindow
# =============================================================================
run_midwindow() {
    echo "########## scenario: midwindow ##########"
    local LAST_STEP=$NUM_TRAIN_BATCHES
    if (( DIE_AT_STEP <= 0 || DIE_AT_STEP >= NUM_TRAIN_BATCHES )); then
        echo "Warning: die_at_step=$DIE_AT_STEP not strictly inside window (0, $NUM_TRAIN_BATCHES)" >&2
    fi
    if (( DIE_AT_STEP % IN_WINDOW_FREQ != 0 )); then
        echo "Warning: die_at_step=$DIE_AT_STEP not a multiple of in_window_freq=$IN_WINDOW_FREQ; no save lands exactly at crash" >&2
    fi

    # P1 baseline
    clean_ckpt "$CKPT_ROOT"
    run_phase baseline \
        "NUM_TRAIN_TS=1" "EVAL_EVERY_N_WINDOWS=0" "METRIC_LOG_FREQ=1" \
        "NUM_TRAIN_BATCHES=$NUM_TRAIN_BATCHES" "NUM_EVAL_BATCHES=$NUM_EVAL_BATCHES" \
        "TRAIN_SPLIT_PERCENTAGE=1.0" "DIE_AT_STEP=-1"
    wait_for_log baseline "PHASE_EXIT=0" "$MW_TIMEOUT"; local rc=$?
    cleanup_workers
    (( rc != 0 )) && { echo "FAIL: midwindow baseline didn't finish (rc=$rc)"; tail -20 "$LOG_DIR/baseline.log"; return 1; }

    # P2 interrupted
    clean_ckpt "$CKPT_ROOT"
    run_phase interrupt \
        "NUM_TRAIN_TS=1" "EVAL_EVERY_N_WINDOWS=0" "METRIC_LOG_FREQ=1" \
        "NUM_TRAIN_BATCHES=$NUM_TRAIN_BATCHES" "NUM_EVAL_BATCHES=$NUM_EVAL_BATCHES" \
        "TRAIN_SPLIT_PERCENTAGE=1.0" \
        "IN_WINDOW_CKPT_FREQ=$IN_WINDOW_FREQ" "KEEP_LAST_N=1" \
        "DIE_AT_STEP=$DIE_AT_STEP" "CKPT_PATH=$CKPT_ROOT"
    wait_for_log interrupt "die_at_step=$DIE_AT_STEP hit" "$MW_TIMEOUT"; rc=$?
    cleanup_workers
    (( rc != 0 )) && { echo "FAIL: interrupt didn't hit die_at_step (rc=$rc)"; tail -20 "$LOG_DIR/interrupt.log"; return 1; }
    echo "Saved checkpoints after interrupt: $(sx "ls '$CKPT_ROOT' 2>/dev/null | tr '\n' ' '")"

    # P3 resume
    run_phase resume \
        "NUM_TRAIN_TS=1" "EVAL_EVERY_N_WINDOWS=0" "METRIC_LOG_FREQ=1" \
        "NUM_TRAIN_BATCHES=$NUM_TRAIN_BATCHES" "NUM_EVAL_BATCHES=$NUM_EVAL_BATCHES" \
        "TRAIN_SPLIT_PERCENTAGE=1.0" \
        "IN_WINDOW_CKPT_FREQ=$IN_WINDOW_FREQ" "KEEP_LAST_N=1" \
        "DIE_AT_STEP=-1" "CKPT_PATH=$CKPT_ROOT"
    # PHASE_EXIT=0 only after the (blocking) end-of-window save renames cleanly,
    # so this also confirms the final atomic save completed.
    wait_for_log resume "PHASE_EXIT=0" "$MW_TIMEOUT"; rc=$?
    cleanup_workers
    (( rc != 0 )) && { echo "FAIL: resume didn't finish (rc=$rc)"; tail -20 "$LOG_DIR/resume.log"; return 1; }

    # HARD functional invariants (deterministic; the real correctness proof).
    if ! grep -qE "Resuming mid-window at train_ts=[0-9]+ batch_idx_in_window=$DIE_AT_STEP\b" "$LOG_DIR/resume.log" 2>/dev/null; then
        fail "resume did not re-enter mid-window at batch_idx_in_window=$DIE_AT_STEP"
        grep -E "Resuming" "$LOG_DIR/resume.log" 2>/dev/null | head -2
    fi
    local rng_restored
    rng_restored=$(grep -c "RNG state restored from" "$LOG_DIR/resume.log" 2>/dev/null || echo 0)
    echo "RNG state restored on $rng_restored ranks"
    (( rng_restored < 1 )) && fail "no RNG state restored on resume"
    local first_resumed
    first_resumed=$(grep -oE 'train - Step [0-9]+ metrics: \{.metric' "$LOG_DIR/resume.log" 2>/dev/null \
        | grep -oE 'Step [0-9]+' | awk '{print $2}' | sort -n | head -1)
    echo "First resumed train step: $first_resumed (expect $((DIE_AT_STEP + 1)))"
    [[ "$first_resumed" != "$((DIE_AT_STEP + 1))" ]] && fail "resume did not continue at step $((DIE_AT_STEP + 1)) (got $first_resumed)"

    # On-disk: atomic save + retention.
    local num_ckpt stale_ckpt
    num_ckpt=$(sx "ls '$CKPT_ROOT' 2>/dev/null | grep -E '^[0-9]+$' | wc -l" | tr -d ' ')
    stale_ckpt=$(sx "ls '$CKPT_ROOT' 2>/dev/null | grep -E '\\.(tmp|old|staging)$' | wc -l" | tr -d ' ')
    echo "Final: $num_ckpt numeric ckpt subdirs, $stale_ckpt stale dirs (expect 1, 0)"
    [[ "$num_ckpt" != "1" ]] && fail "keep_last_n=1 violated (got $num_ckpt)"
    [[ "$stale_ckpt" != "0" ]] && fail "stale .tmp/.old/.staging dirs left behind ($stale_ckpt)"

    # Trajectory closeness (loose sanity bound, NOT bit-equality).
    python3 "$PYHELPER" parse "$LOG_DIR/baseline.log" "$LOG_DIR/traj_baseline.json"
    python3 "$PYHELPER" parse "$LOG_DIR/resume.log" "$LOG_DIR/traj_resumed.json"
    if ! python3 "$PYHELPER" compare "$LOG_DIR/traj_baseline.json" "$LOG_DIR/traj_resumed.json" \
            --min-resume-step $((DIE_AT_STEP + 1)) --atol "$ATOL"; then
        fail "trajectory diverged beyond $ATOL (likely wrong data slice / unrestored state)"
    fi
    (( FAIL == 0 )) && echo "=== midwindow: PASS ===" || echo "=== midwindow: FAIL ==="
}

# =============================================================================
# SCENARIO: multiwindow  (regression guard for the broadcast + barrier fixes)
# =============================================================================
# Common split contract — MUST be byte-identical between mw_seed and mw_resume,
# else the resume aborts on a split-contract mismatch (the holdout_ts default of
# start_ts+num_train_ts differs between a 1-window seed and an MW_TS resume, so
# it is PINNED here).
MW_SPLIT_ENV=( "TRAIN_SPLIT_PERCENTAGE=$MW_SPLIT" "SPLIT_SALT=0"
               "EVAL_HOLDOUT_TS=$MW_HOLDOUT_TS" "EVAL_HOLDOUT_NUM_WINDOWS=1" )

run_multiwindow() {
    echo "########## scenario: multiwindow ##########"
    local sum

    # P1 mw_baseline — cold multi-window run with data-pct eval.
    clean_ckpt "$CKPT_ROOT"
    run_phase mw_baseline \
        "NUM_TRAIN_TS=$MW_TS" "NUM_TRAIN_BATCHES=$MW_BATCHES" "NUM_EVAL_BATCHES=$MW_EVAL_BATCHES" \
        "EVAL_EVERY_N_WINDOWS=0" "EVAL_EVERY_DATA_PCT=$MW_EVAL_PCT" "METRIC_LOG_FREQ=1" \
        "${MW_SPLIT_ENV[@]}"
    wait_for_log mw_baseline "PHASE_EXIT=" "$MW_RUN_TIMEOUT"; local rc=$?
    cleanup_workers
    (( rc == 1 )) && { echo "FAIL: mw_baseline crashed"; tail -30 "$LOG_DIR/mw_baseline.log"; return 1; }
    (( rc == 2 )) && { echo "FAIL: mw_baseline timed out (possible boundary deadlock)"; tail -30 "$LOG_DIR/mw_baseline.log"; return 1; }

    sum="$LOG_DIR/mw_baseline.summary.json"
    python3 "$PYHELPER" summarize "$LOG_DIR/mw_baseline.log" "$sum" >/dev/null
    echo "--- mw_baseline summary ---"; cat "$sum"
    local exit_code anchors barriers dpct_setup dpct_trig
    exit_code=$(jget "$sum" phase_exit)
    anchors=$(jget "$sum" total_train_anchors_calls)
    barriers=$(jget "$sum" window_barrier_count)
    dpct_setup=$(jget "$sum" data_pct_eval_setup)
    dpct_trig=$(jget "$sum" data_pct_eval_trigger_count)
    # (barrier B) ran through ALL windows and exited 0 — no boundary deadlock.
    [[ "$exit_code" != "0" ]] && fail "mw_baseline did not complete cleanly (phase_exit=$exit_code)"
    [[ "$barriers" != "$MW_TS" ]] && fail "window barrier fired $barriers times, expected $MW_TS (one per window; need world_size>=2)"
    # (broadcast A) total_train_anchors computed exactly once (rank 0), not Nx.
    # It is computed at loop SETUP (before any training), so this exercises the
    # broadcast regardless of whether an eval later fires.
    [[ "$anchors" != "1" ]] && fail "total_train_anchors computed $anchors times, expected 1 (rank-0 broadcast regressed)"
    # data-fraction eval cadence set up (the path that needs total_train_anchors).
    [[ "$dpct_setup" != "True" ]] && fail "data-pct eval cadence not set up (total_train_anchors path not reached)"
    # Trigger firing depends on (full-window) anchor count vs the few test steps,
    # so it is informational — not required to exercise the broadcast fix.
    echo "data-pct eval triggers fired: $dpct_trig (informational)"

    # P2 mw_seed — 1 window → clean WINDOW_COMPLETE checkpoint.
    clean_ckpt "$CKPT_ROOT"
    run_phase mw_seed \
        "NUM_TRAIN_TS=1" "NUM_TRAIN_BATCHES=$MW_BATCHES" "NUM_EVAL_BATCHES=$MW_EVAL_BATCHES" \
        "EVAL_EVERY_N_WINDOWS=0" "EVAL_EVERY_DATA_PCT=$MW_EVAL_PCT" "METRIC_LOG_FREQ=1" \
        "KEEP_LAST_N=1" "CKPT_PATH=$CKPT_ROOT" "${MW_SPLIT_ENV[@]}"
    wait_for_log mw_seed "PHASE_EXIT=0" "$MW_RUN_TIMEOUT"; rc=$?
    cleanup_workers
    (( rc != 0 )) && { echo "FAIL: mw_seed didn't finish/checkpoint (rc=$rc)"; tail -30 "$LOG_DIR/mw_seed.log"; return 1; }
    local seed_ckpt
    seed_ckpt=$(sx "ls '$CKPT_ROOT' 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1" | tr -d ' ')
    echo "mw_seed end-of-window checkpoint: ${seed_ckpt:-<none>} (expect $START_TS)"
    [[ "$seed_ckpt" != "$START_TS" ]] && { fail "mw_seed did not save end-of-window ckpt $START_TS (got '$seed_ckpt')"; return 1; }

    # P3 mw_resume — relaunch over MW_TS windows; resume past the completed
    # window and CROSS the boundary into the remaining windows (the exact case
    # that used to deadlock). The full split contract matches the seed.
    run_phase mw_resume \
        "NUM_TRAIN_TS=$MW_TS" "NUM_TRAIN_BATCHES=$MW_BATCHES" "NUM_EVAL_BATCHES=$MW_EVAL_BATCHES" \
        "EVAL_EVERY_N_WINDOWS=0" "EVAL_EVERY_DATA_PCT=$MW_EVAL_PCT" "METRIC_LOG_FREQ=1" \
        "KEEP_LAST_N=1" "CKPT_PATH=$CKPT_ROOT" "${MW_SPLIT_ENV[@]}"
    wait_for_log mw_resume "PHASE_EXIT=" "$MW_RUN_TIMEOUT"; rc=$?
    cleanup_workers
    (( rc == 1 )) && { echo "FAIL: mw_resume crashed"; tail -30 "$LOG_DIR/mw_resume.log"; return 1; }
    (( rc == 2 )) && { echo "FAIL: mw_resume timed out (possible boundary deadlock on resume)"; tail -30 "$LOG_DIR/mw_resume.log"; return 1; }

    sum="$LOG_DIR/mw_resume.summary.json"
    python3 "$PYHELPER" summarize "$LOG_DIR/mw_resume.log" "$sum" >/dev/null
    echo "--- mw_resume summary ---"; cat "$sum"
    local r_exit r_resume_ts r_anchors r_barriers r_dpct
    r_exit=$(jget "$sum" phase_exit)
    r_resume_ts=$(jget "$sum" resume_completed_ts)
    r_anchors=$(jget "$sum" total_train_anchors_calls)
    r_barriers=$(jget "$sum" window_barrier_count)
    r_dpct=$(jget "$sum" data_pct_eval_setup)
    # Resumed from the completed seed window (advanced past the boundary).
    [[ "$r_resume_ts" != "$START_TS" ]] && fail "mw_resume did not resume from completed train_ts=$START_TS (got $r_resume_ts)"
    # Crossed the boundary into the remaining MW_TS-1 windows and exited 0.
    [[ "$r_exit" != "0" ]] && fail "mw_resume did not complete cleanly (phase_exit=$r_exit) — boundary deadlock on resume?"
    [[ "$r_barriers" != "$((MW_TS - 1))" ]] && fail "mw_resume barrier fired $r_barriers times, expected $((MW_TS - 1)) (windows after the resumed one)"
    # Broadcast still once on the resume path; data-pct cadence rebuilt.
    [[ "$r_anchors" != "1" ]] && fail "mw_resume total_train_anchors computed $r_anchors times, expected 1"
    [[ "$r_dpct" != "True" ]] && fail "mw_resume data-pct eval cadence not set up"

    (( FAIL == 0 )) && echo "=== multiwindow: PASS ===" || echo "=== multiwindow: FAIL ==="
}

# =============================================================================
[[ "$SCENARIO" == "midwindow"  || "$SCENARIO" == "all" ]] && run_midwindow
[[ "$SCENARIO" == "multiwindow" || "$SCENARIO" == "all" ]] && run_multiwindow

if [[ "$KEEP" != "1" ]]; then
    rm -rf "$LOG_DIR"
    clean_ckpt "$CKPT_ROOT"
fi

if (( FAIL == 0 )); then
    echo "=== PASS: all selected scenarios validated ==="
    exit 0
fi
echo "=== FAIL: one or more scenarios failed (see above) ==="
exit 1
