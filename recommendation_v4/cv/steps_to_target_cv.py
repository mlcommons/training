#!/usr/bin/env python3
"""Run-to-run convergence CV via the steps-to-target method.

For each target eval-window AUC t, for each run find the first samples_count
where AUC >= t (first-crossing, no interpolation). Report mean/std/CV/min/max
across the runs that reached t.

Reads MLPerf ``:::MLLOG`` logs (``eval_accuracy`` == window_auc/listen_plus).

Examples::

    # auto sweep 0.60 .. max-AUC @ step 0.005
    python3 cv/steps_to_target_cv.py --log-dir /path/to/logs -o cv/table.csv

    # single target
    python3 cv/steps_to_target_cv.py --log-dir /path/to/logs --target 0.75

    # explicit list
    python3 cv/steps_to_target_cv.py --log-dir /path/to/logs --targets 0.70,0.75,0.78

    # range with step
    python3 cv/steps_to_target_cv.py --log-dir /path/to/logs --target-range 0.60:0.78 --step 0.005
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_MLLOG = re.compile(r":::MLLOG (\{.*\})\s*$")


def parse_log(path):
    """MLPerf log -> (samples_sorted, auc_sorted). Empty arrays if no evals."""
    pts = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = _MLLOG.search(line)
            if not m:
                continue
            try:
                rec = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if rec.get("key") != "eval_accuracy" or rec.get("value") is None:
                continue
            sc = rec.get("metadata", {}).get("samples_count")
            if sc is not None:
                pts[int(sc)] = float(rec["value"])
    if not pts:
        return np.array([]), np.array([])
    samples = np.array(sorted(pts))
    auc = np.array([pts[s] for s in samples])
    return samples, auc


def load_runs(log_dir):
    """All *.log files in log_dir -> {basename: (samples, auc)}."""
    paths = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not paths:
        sys.exit(f"no *.log files in {log_dir}")
    runs = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        samples, auc = parse_log(path)
        if len(auc) >= 1:
            runs[name] = (samples, auc)
        print(f"  {name}: {len(auc)} eval points", file=sys.stderr)
    return runs


def first_crossing(samples, auc, target):
    for s, a in zip(samples, auc):
        if a >= target:
            return s
    return None


def resolve_targets(args, runs):
    """Build target AUC list from CLI flags."""
    if args.target is not None:
        return [float(args.target)]
    if args.targets is not None:
        return [float(t.strip()) for t in args.targets.split(",") if t.strip()]

    lo, hi = args.target_range
    top = max(a.max() for _, a in runs.values())
    hi = min(hi, top)
    if lo > hi:
        sys.exit(f"target-range lo={lo} > hi={hi} (max AUC across runs = {top:.4f})")
    return list(np.arange(lo, hi + 1e-9, args.step))


def cv_table(runs, targets, min_runs):
    rows = []
    for t in targets:
        hits = [s for samples, auc in runs.values()
                if (s := first_crossing(samples, auc, t)) is not None]
        n = len(hits)
        if n < min_runs:
            continue
        arr = np.array(hits, dtype=float)
        mean = arr.mean()
        std = arr.std(ddof=1)
        rows.append({
            "target_eval_auc": round(float(t), 4),
            "runs_reached": n,
            "samples_mean": round(mean, 1),
            "samples_std": round(std, 1),
            "cv_pct": round(std / mean * 100.0, 3) if mean > 0 else 0.0,
            "cv_fraction": round(std / mean, 5) if mean > 0 else 0.0,
            "samples_min": int(arr.min()),
            "samples_max": int(arr.max()),
        })
    return pd.DataFrame(rows)


def parse_target_range(s):
    if ":" not in s:
        raise argparse.ArgumentTypeError("expected LO:HI, e.g. 0.60:0.78")
    lo, hi = s.split(":", 1)
    return float(lo), float(hi)


def main():
    p = argparse.ArgumentParser(
        description="Steps-to-target CV table from a folder of MLPerf logs.")
    p.add_argument("--log-dir", required=True,
                   help="directory containing one *.log per seed run")
    p.add_argument("-o", "--output",
                   help="output CSV path (default: <log-dir>/steps_to_target_cv.csv)")
    p.add_argument("--min-runs", type=int, default=2,
                   help="min runs that must reach a target to include it (default: 2)")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--target", type=float,
                   help="single target eval-window AUC, e.g. 0.75")
    g.add_argument("--targets", metavar="T1,T2,...",
                   help="comma-separated target AUCs, e.g. 0.70,0.75,0.78")
    g.add_argument("--target-range", type=parse_target_range, metavar="LO:HI",
                   help="inclusive AUC range, e.g. 0.60:0.78 (default when unset)")

    p.add_argument("--step", type=float, default=0.005,
                   help="step for --target-range sweep (default: 0.005)")
    p.add_argument("--range-lo", type=float, default=0.60,
                   help="lower bound when no target flag given (default: 0.60)")
    args = p.parse_args()

    if args.target_range is None and args.target is None and args.targets is None:
        args.target_range = (args.range_lo, 1.0)  # hi clipped to max AUC in resolve_targets

    print(f"loading logs from {args.log_dir}", file=sys.stderr)
    runs = load_runs(args.log_dir)
    if len(runs) < args.min_runs:
        sys.exit(f"only {len(runs)} usable run(s) in {args.log_dir} (need >= {args.min_runs})")

    targets = resolve_targets(args, runs)
    df = cv_table(runs, targets, args.min_runs)
    if df.empty:
        sys.exit("no targets had enough runs — check --target / --target-range / --min-runs")

    out = args.output or os.path.join(args.log_dir, "steps_to_target_cv.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(runs)} runs, {len(df)} targets)", file=sys.stderr)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
