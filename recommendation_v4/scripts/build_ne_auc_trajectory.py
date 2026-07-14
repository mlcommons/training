#!/usr/bin/env python3
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

"""Build a combined train+eval NE/AUC trajectory from a streaming-train-eval log.

The streaming loop (generative_recommenders/dlrm_v3/train/utils.py) emits, via
MetricsLogger.compute(), one line per logged step of the form:

  INFO:utils:train - Step 201 metrics: {'metric/lifetime_ne/listen_plus':
    tensor(1.0182, dtype=torch.float64), 'metric/window_ne/listen_plus':
    tensor(0.9846, ...), ..., 'metric/window_auc/listen_plus': tensor(0.5912),
    'metric/lifetime_auc/listen_plus': tensor(0.5480)}

and the analogous `eval - Step N metrics:` lines during each (full-holdout) eval
window, plus throughput lines:

  INFO:utils:train - Step 201 perf: local_sps=97.0 global_sps=776.2
    step_ms=10553.89 elapsed_sec=680.6 total_samples=205824

This script parses all three, for a chosen task (default listen_plus), and writes:
  * <out>/trajectory.json   — {"train": {step: {...}}, "eval": {...}, "perf": [...]}
  * <out>/trajectory.csv    — long-form rows (mode, step, metric, value)
  * <out>/trajectory_ne_auc.png — NE and AUC vs train step, train + eval overlaid
                                  (skipped gracefully if matplotlib is absent)

It is dependency-light (stdlib + optional matplotlib) so it runs anywhere the
log is readable, including the head node.

Usage:
  python3 scripts/build_ne_auc_trajectory.py LOG [--out DIR] [--task listen_plus]
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# `train - Step 201 metrics: {...}` / `eval - Step 17 metrics: {...}`
_STEP_RE = re.compile(r"(train|eval) - Step (\d+) metrics: \{(.*)\}")
# `metric/<prefix><name>/<task>': tensor(<value>` — value may be int/float/sci.
_METRIC_RE = re.compile(
    r"metric/([A-Za-z0-9_]+)/([A-Za-z0-9_+]+)'?\s*:\s*tensor\(\s*([-0-9.eE+]+)"
)
# `train - Step 201 perf: local_sps=97.0 global_sps=776.2 step_ms=10553.89 `
# `elapsed_sec=680.6 total_samples=205824`
_PERF_RE = re.compile(
    r"train - Step (\d+) perf: local_sps=([-0-9.eE+]+) global_sps=([-0-9.eE+]+) "
    r"step_ms=([-0-9.eE+]+) elapsed_sec=([-0-9.eE+]+) total_samples=(\d+)"
)
# `[boundary] eval_ts=181 eval first-batch ...` — marks the start of a full-holdout
# eval block; the eval runs at whatever the latest train global step was, so we use
# it to anchor each eval's metrics onto the shared train-global-step x-axis.
_EVAL_BOUNDARY_RE = re.compile(r"\[boundary\] eval_ts=(\d+) eval first-batch")

# Metrics we surface in the trajectory (others are still captured if present).
_KEEP = ("window_ne", "lifetime_ne", "window_auc", "lifetime_auc",
         "window_accuracy", "lifetime_accuracy", "window_gauc", "lifetime_gauc")


def _parse_metrics(body: str, task: str) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for name, tname, val in _METRIC_RE.findall(body):
        if tname != task:
            continue
        try:
            row[name] = float(val)
        except ValueError:
            continue
    return row


def parse_log(
    log_path: str, task: str
) -> Tuple[Dict[str, Dict[int, Dict[str, float]]], List[Dict[str, float]]]:
    """Return ({'train': {step: {metric: val}}, 'eval': {...}}, perf_rows).

    Train is keyed by train global step (last write wins — duplicate per-rank
    prints are identical). Eval uses a per-rank-resetting internal step counter
    that restarts every eval window, so we instead anchor each eval window onto
    the *train global step at which it ran* (the loop trains window T then evals
    window T+1, so the eval's anchor is the last train step before it). Each eval
    window collapses to a single point carrying its final, most-aggregated
    full-holdout metrics, plus `eval_window` (the eval_ts) for reference.
    """
    out: Dict[str, Dict[int, Dict[str, float]]] = {"train": {}, "eval": {}}
    perf: List[Dict[str, float]] = []

    last_train_step = 0
    cur_anchor: Optional[int] = None   # train global step this eval block runs at
    cur_ts: Optional[int] = None       # eval window id (eval_ts)
    cur_row: Optional[Dict[str, float]] = None  # final row of the current block
    cur_internal: Optional[int] = None  # last eval internal step (reset detection)

    def flush_eval() -> None:
        nonlocal cur_anchor, cur_ts, cur_row, cur_internal
        if cur_row:
            anchor = cur_anchor if cur_anchor is not None else last_train_step
            row = dict(cur_row)
            if cur_ts is not None:
                row["eval_window"] = float(cur_ts)
            key = anchor
            while key in out["eval"]:  # keep distinct evals from colliding
                key += 1
            out["eval"][key] = row
        cur_anchor = cur_ts = cur_row = cur_internal = None

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            pm = _PERF_RE.search(line)
            if pm:
                perf.append({
                    "step": int(pm.group(1)),
                    "local_sps": float(pm.group(2)),
                    "global_sps": float(pm.group(3)),
                    "step_ms": float(pm.group(4)),
                    "elapsed_sec": float(pm.group(5)),
                    "total_samples": int(pm.group(6)),
                })
                continue
            bm = _EVAL_BOUNDARY_RE.search(line)
            if bm:
                # The boundary line (a different logger) can interleave before OR
                # after this eval's metric lines, so don't use it to delimit the
                # block — just tag the current block with its eval_ts. Block
                # boundaries come from eval-step resets / training resuming.
                if cur_anchor is None:
                    cur_anchor = last_train_step
                cur_ts = int(bm.group(1))
                continue
            m = _STEP_RE.search(line)
            if not m:
                continue
            mode, step_s, body = m.group(1), m.group(2), m.group(3)
            step = int(step_s)
            row = _parse_metrics(body, task)
            if mode == "train":
                last_train_step = step
                if cur_anchor is not None or cur_row is not None:
                    flush_eval()  # an eval block ends when training resumes
                if row:
                    out["train"][step] = row  # last write wins
            else:  # eval — accumulate into the current block (last = most aggregated)
                # Fallback for logs without a boundary marker: a drop in the eval
                # internal step counter signals a fresh eval window.
                if (cur_internal is not None and step < cur_internal
                        and cur_anchor is None):
                    flush_eval()
                if cur_anchor is None:
                    cur_anchor = last_train_step
                cur_internal = step
                if row:
                    cur_row = row
    flush_eval()
    return out, perf


def write_outputs(
    traj: Dict[str, Dict[int, Dict[str, float]]],
    perf: List[Dict[str, float]],
    out_dir: str,
    task: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "trajectory.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "task": task,
                "train": {str(k): v for k, v in sorted(traj["train"].items())},
                "eval": {str(k): v for k, v in sorted(traj["eval"].items())},
                "perf": perf,
            },
            f,
            indent=2,
        )

    csv_path = os.path.join(out_dir, "trajectory.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "step", "metric", "value"])
        for mode in ("train", "eval"):
            for step in sorted(traj[mode]):
                for metric, val in traj[mode][step].items():
                    w.writerow([mode, step, metric, val])

    n_train = len(traj["train"])
    n_eval = len(traj["eval"])
    print(f"Parsed {n_train} train points, {n_eval} eval points, "
          f"{len(perf)} perf points (task={task}).", file=sys.stderr)
    print(f"Wrote {json_path}", file=sys.stderr)
    print(f"Wrote {csv_path}", file=sys.stderr)

    _maybe_plot(traj, out_dir, task)


def _maybe_plot(
    traj: Dict[str, Dict[int, Dict[str, float]]], out_dir: str, task: str
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"matplotlib unavailable ({e}); skipping plot.", file=sys.stderr)
        return

    def series(mode: str, metric: str) -> Tuple[List[int], List[float]]:
        steps = sorted(s for s in traj[mode] if metric in traj[mode][s])
        return steps, [traj[mode][s][metric] for s in steps]

    fig, (ax_ne, ax_auc) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for metric, style in (("window_ne", "-"), ("lifetime_ne", "--")):
        xs, ys = series("train", metric)
        if xs:
            ax_ne.plot(xs, ys, style, label=f"train/{metric}", alpha=0.85)
    for metric, marker in (("window_ne", "o"), ("lifetime_ne", "s")):
        xs, ys = series("eval", metric)
        if xs:
            ax_ne.plot(xs, ys, marker=marker, ms=5, ls="-", lw=1.0, alpha=0.9,
                       label=f"eval/{metric}")
    ax_ne.set_ylabel("NE (normalized entropy)")
    ax_ne.set_title(f"yambda-5b streaming train+eval trajectory — task={task}")
    ax_ne.grid(True, alpha=0.3)
    ax_ne.legend(fontsize=8, ncol=2)

    for metric, style in (("window_auc", "-"), ("lifetime_auc", "--")):
        xs, ys = series("train", metric)
        if xs:
            ax_auc.plot(xs, ys, style, label=f"train/{metric}", alpha=0.85)
    for metric, marker in (("window_auc", "o"), ("lifetime_auc", "s")):
        xs, ys = series("eval", metric)
        if xs:
            ax_auc.plot(xs, ys, marker=marker, ms=5, ls="-", lw=1.0, alpha=0.9,
                        label=f"eval/{metric}")
    ax_auc.set_ylabel("AUC")
    ax_auc.set_xlabel("train global step (eval points anchored to the step they ran at)")
    ax_auc.grid(True, alpha=0.3)
    ax_auc.legend(fontsize=8, ncol=2)

    png_path = os.path.join(out_dir, "trajectory_ne_auc.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"Wrote {png_path}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="Path to the streaming train.log")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: <log_dir>/<log_stem>_trajectory)")
    ap.add_argument("--task", default="listen_plus",
                    help="Task name to extract (default: listen_plus)")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"Log not found: {args.log}", file=sys.stderr)
        return 2
    out_dir = args.out
    if out_dir is None:
        stem = os.path.splitext(os.path.basename(args.log))[0]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.log)),
                               f"{stem}_trajectory")

    traj, perf = parse_log(args.log, args.task)
    write_outputs(traj, perf, out_dir, args.task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
