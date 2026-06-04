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

Validates the four resume features end-to-end on the yambda-5b stack:
  1. Mid-window save (in_window_checkpoint_frequency)
  2. Within-window exact-once skip (StreamingWindowSampler.set_window skip)
  3. Auto-detect-latest checkpoint subdir
  4. keep_last_n retention (default 1)

Test flow (driven by `scripts/streaming_resume_test.sh`):
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
from typing import Dict, Tuple

# Per-step metrics from MetricsLogger.compute_and_log are emitted like:
#   "train - Step 51 metrics: {'metric/lifetime_ne/listen_plus': tensor(1.0954, ...)
#     'metric/window_ne/listen_plus': tensor(0.9940, ...),
#     'metric/window_accuracy/listen_plus': tensor(0.6231, ...) ..."
_STEP_RE = re.compile(r"train - Step (\d+) metrics:")
_WNE_RE = re.compile(r"window_ne/listen_plus.*?tensor\(([0-9.]+)")
_WAUC_RE = re.compile(r"window_auc/listen_plus.*?tensor\(([0-9.]+)")
_WACC_RE = re.compile(r"window_accuracy/listen_plus.*?tensor\(([0-9.]+)")


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
