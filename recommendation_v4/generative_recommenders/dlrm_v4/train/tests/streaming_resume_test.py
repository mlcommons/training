# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end failure-injection test for streaming resume.

Two scenarios, driven by the sibling `streaming_resume_test.sh` (see its header
for the full, platform-general launch wiring — NVIDIA B200 and AMD MI350/MI355).
This module is the shared log parser + a CLI the driver shells out to.

SCENARIO `midwindow` — exact-once mid-window resume. Validates the four
single-window resume features end-to-end on the yambda-5b stack:
  1. Mid-window save (in_window_checkpoint_frequency)
  2. Within-window exact-once skip (StreamingWindowSampler.set_window skip)
  3. Auto-detect-latest checkpoint subdir
  4. keep_last_n retention (default 1)

SCENARIO `multiwindow` — distributed-sync regression guard for the two fixes the
mid-window test cannot reach (it runs ONE window with per-window eval off):
  A. total_train_anchors() computed once on rank 0 + broadcast (not world_size×).
  B. window-boundary dist.barrier() before the first forward of each window.
Both only matter across >=2 windows with the data-fraction eval cadence
(EVAL_EVERY_DATA_PCT>0) active, and the deadlock they fix originally struck at a
window boundary mid-run — so the scenario trains multiple windows AND resumes
across a completed-window boundary. The signals are extracted by `summarize`
(see `summarize_run`) and asserted in the shell driver.

Test flow (driven by the sibling `streaming_resume_test.sh`):
  Phase 1 (baseline): Run streaming-train-eval for N=2 train_ts × K batches/window
    with die_at_step=-1. Capture per-batch window_ne / window_auc into traj_baseline.json.
  Phase 2 (interrupt): Same config but die_at_step=M (M mid-window-2). Expect
    process to exit(42) after the in-window checkpoint at step M lands.
  Phase 3 (resume): Re-launch with same CKPT_PATH (auto-latest picks the
    in-window save). Continue to the same total step count. Capture
    traj_resumed.json (which only contains the post-resume steps).

  Correctness is proven by the FUNCTIONAL INVARIANTS checked in the shell
  driver (resumed at exactly batch_idx_in_window, per-rank RNG restored, atomic
  save + keep_last_n), NOT by bit-equal trajectory matching. The training stack
  is nondeterministic across runs (non-deterministic atomic scatter-add in the
  embedding/attention backward on ROCm): two independent *cold* runs already
  drift ~7e-4 in window_ne over 20 steps, and early-training chaos amplifies
  it, so resume-vs-baseline can differ by a few percent even when resume is
  perfect. The trajectory comparison here is therefore a LOOSE closeness bound
  (default atol below) that only flags gross divergence — wrong data slice or
  unrestored model state — while tolerating nondeterministic drift.

This module also provides a CLI entry point used by the shell driver to (a)
parse a train.log into a step-keyed dict of metrics, and (b) compare two such
dicts and fail loudly on mismatch.
"""

import argparse
import json
import re
import sys
from typing import Dict, List, Optional, Tuple

# Per-step metrics from MetricsLogger.compute_and_log are emitted like:
#   "train - Step 51 metrics: {'metric/lifetime_ne/listen_plus': tensor(1.0954, ...)
#     'metric/window_ne/listen_plus': tensor(0.9940, ...),
#     'metric/window_accuracy/listen_plus': tensor(0.6231, ...) ..."
_STEP_RE = re.compile(r"train - Step (\d+) metrics:")
_WNE_RE = re.compile(r"window_ne/listen_plus.*?tensor\(([0-9.]+)")
_WAUC_RE = re.compile(r"window_auc/listen_plus.*?tensor\(([0-9.]+)")
_WACC_RE = re.compile(r"window_accuracy/listen_plus.*?tensor\(([0-9.]+)")


# --- multi-window / data-pct-eval regression signals -------------------------
# These cover the two distributed-sync fixes that the single-window mid-window
# test above does NOT exercise (it runs one window with per-window eval off):
#
#   (A) total_train_anchors() rank-0 broadcast. The data-fraction eval cadence
#       needs total_train_anchors — a multi-minute, single-threaded O(N) gather
#       + uid-hash over the mmap'd anchor array. Run on EVERY rank it both wastes
#       8x CPU and desyncs the NCCL stream (a fast rank races into the first
#       embedding all-to-all while slow ranks still hash) → deadlock. The fix
#       computes it ONCE on rank 0 and broadcasts the scalar. yambda logs exactly
#       one `total_train_anchors(start_ts=…)` line per call, so the regression
#       guard is: that line appears EXACTLY ONCE per launch (was world_size×).
#
#   (B) window-boundary barrier. Per-window data prep (`window_indices`, an O(N)
#       mask over the ~18GB mmap) finishes at very different times across ranks;
#       without a sync before the first forward the collective stream desyncs and
#       the job hangs at the boundary. The fix adds a dist.barrier() at each
#       window boundary. It is silent on the healthy path, so the trainer emits a
#       `[window-barrier] … rendezvous complete` line (rank 0) per crossed window
#       ONLY under WINDOW_BARRIER_DEBUG=1 — the guard counts those == #windows.
_TTA_RE = re.compile(r"total_train_anchors\(start_ts=(\d+),\s*num_ts=(\d+)\):")
_BARRIER_RE = re.compile(r"\[window-barrier\] train_ts=(\d+) rendezvous complete")
_DATA_PCT_SETUP_RE = re.compile(
    r"\[data-pct-eval\] eval_every_data_pct=.*?eval_interval_steps=(\d+)"
)
_DATA_PCT_TRIGGER_RE = re.compile(r"\[data-pct-eval\] trigger eval train_ts=(\d+)")
_RESUME_COMPLETED_RE = re.compile(r"Resuming from completed train_ts=(\d+)")
_RESUME_MIDWINDOW_RE = re.compile(
    r"Resuming mid-window at train_ts=(\d+) batch_idx_in_window=(\d+)"
)
# Test driver appends this sentinel after the trainer returns (clean OR crash);
# code 0 == the run finished all requested windows + final eval without hanging.
_PHASE_EXIT_RE = re.compile(r"PHASE_EXIT=(-?\d+)")


def summarize_run(log_path: str) -> Dict[str, object]:
    """Extract the multi-window / data-pct-eval regression signals from a run log.

    Returns a JSON-able dict the shell driver asserts on. All counts are over the
    WHOLE log (one launch's worth — the driver uses a fresh per-phase log)."""
    tta_calls: List[Tuple[int, int]] = []
    barrier_windows: List[int] = []
    data_pct_eval_setup: bool = False
    data_pct_eval_interval: Optional[int] = None
    data_pct_eval_triggers: List[int] = []
    resume_completed_ts: Optional[int] = None
    resume_midwindow: Optional[Tuple[int, int]] = None
    phase_exit: Optional[int] = None
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = _TTA_RE.search(line)
            if m:
                tta_calls.append((int(m.group(1)), int(m.group(2))))
            m = _BARRIER_RE.search(line)
            if m:
                barrier_windows.append(int(m.group(1)))
            m = _DATA_PCT_SETUP_RE.search(line)
            if m:
                data_pct_eval_setup = True
                data_pct_eval_interval = int(m.group(1))
            m = _DATA_PCT_TRIGGER_RE.search(line)
            if m:
                data_pct_eval_triggers.append(int(m.group(1)))
            m = _RESUME_COMPLETED_RE.search(line)
            if m:
                resume_completed_ts = int(m.group(1))
            m = _RESUME_MIDWINDOW_RE.search(line)
            if m:
                resume_midwindow = (int(m.group(1)), int(m.group(2)))
            m = _PHASE_EXIT_RE.search(line)
            if m:
                phase_exit = int(m.group(1))
    return {
        # (A) rank-0 broadcast: must be exactly 1 (was world_size× before the fix)
        "total_train_anchors_calls": len(tta_calls),
        "total_train_anchors_args": tta_calls,
        # (B) barrier executed once per crossed window (rank 0, debug-gated)
        "window_barrier_count": len(barrier_windows),
        "windows_trained": sorted(set(barrier_windows)),
        # data-fraction eval cadence active + actually fired
        "data_pct_eval_setup": data_pct_eval_setup,
        "data_pct_eval_interval_steps": data_pct_eval_interval,
        "data_pct_eval_trigger_count": len(data_pct_eval_triggers),
        # resume classification
        "resume_completed_ts": resume_completed_ts,
        "resume_midwindow": resume_midwindow,
        # terminal status (None => still running / killed without sentinel)
        "phase_exit": phase_exit,
    }


def parse_trajectory(log_path: str) -> Dict[int, Dict[str, float]]:
    """Extract a {step: {window_ne, window_auc, window_accuracy}} dict from a
    train.log. The grep is loose on the metric line itself — we accept the
    very long truncated form MetricsLogger prints."""
    out: Dict[int, Dict[str, float]] = {}
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = _STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            wne = _WNE_RE.search(line)
            wauc = _WAUC_RE.search(line)
            wacc = _WACC_RE.search(line)
            if not (wne and wauc and wacc):
                continue
            # Only keep ONE entry per step — log can have duplicate per-rank
            # prints; first one wins (they're identical).
            if step in out:
                continue
            out[step] = {
                "window_ne": float(wne.group(1)),
                "window_auc": float(wauc.group(1)),
                "window_accuracy": float(wacc.group(1)),
            }
    return out


def compare_trajectories(
    baseline: Dict[int, Dict[str, float]],
    resumed: Dict[int, Dict[str, float]],
    min_resume_step: int,
    atol: float = 0.15,
) -> Tuple[bool, str]:
    """Compare baseline vs resumed trajectories for steps >= min_resume_step.

    This is a LOOSE closeness bound, not a bit-equality check — see the module
    docstring. `atol` defaults to a value that tolerates the nondeterministic
    cross-run drift of this stack while still catching gross resume bugs.
    Returns (ok, message). `ok=False` on any divergence outside `atol`."""
    steps = sorted(s for s in resumed if s >= min_resume_step)
    if not steps:
        return False, f"No resumed steps >= {min_resume_step}"
    mismatches = []
    for s in steps:
        if s not in baseline:
            mismatches.append(f"step {s}: missing from baseline")
            continue
        b = baseline[s]
        r = resumed[s]
        for key in ("window_ne", "window_auc", "window_accuracy"):
            if abs(b[key] - r[key]) > atol:
                mismatches.append(
                    f"step {s} {key}: baseline={b[key]:.6f} "
                    f"resumed={r[key]:.6f} diff={b[key]-r[key]:+.6f}"
                )
    if mismatches:
        return False, (
            f"{len(mismatches)} mismatches across {len(steps)} resumed steps "
            f"(atol={atol}):\n  " + "\n  ".join(mismatches[:10])
        )
    return True, (
        f"{len(steps)} resumed steps match baseline within atol={atol} "
        f"(range: step {steps[0]}..{steps[-1]})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse", help="Parse a train.log → traj JSON")
    p_parse.add_argument("log")
    p_parse.add_argument("out")

    p_cmp = sub.add_parser("compare", help="Compare baseline vs resumed traj JSONs")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument("resumed")
    p_cmp.add_argument("--min-resume-step", type=int, required=True)
    p_cmp.add_argument("--atol", type=float, default=0.15)

    p_sum = sub.add_parser(
        "summarize",
        help="Emit multi-window / data-pct-eval regression signals from a run log",
    )
    p_sum.add_argument("log")
    p_sum.add_argument("out", nargs="?", help="optional JSON output path")

    args = ap.parse_args()
    if args.cmd == "parse":
        traj = parse_trajectory(args.log)
        with open(args.out, "w") as f:
            json.dump(traj, f, indent=2)
        print(f"Wrote {len(traj)} step entries to {args.out}", file=sys.stderr)
        return 0
    if args.cmd == "compare":
        with open(args.baseline) as f:
            baseline = {int(k): v for k, v in json.load(f).items()}
        with open(args.resumed) as f:
            resumed = {int(k): v for k, v in json.load(f).items()}
        ok, msg = compare_trajectories(
            baseline, resumed, args.min_resume_step, atol=args.atol
        )
        print(msg)
        return 0 if ok else 1
    if args.cmd == "summarize":
        summary = summarize_run(args.log)
        out = json.dumps(summary, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(out)
        print(out)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
