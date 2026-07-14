#!/usr/bin/env python3
"""Stitch per-rank Chrome traces from a dlrm_v3 run into one merged file.

When ``Profiler`` runs on multiple ranks, each rank writes its own file:

    <trace_dir>/trace_step{step}_rank{rank}.json

Each per-rank trace uses overlapping ``pid`` namespaces (CPU pid = OS pid;
GPU streams pid = 0..N), so concatenating the raw event lists would collapse
multiple ranks onto the same Perfetto track. This script:

* Identifies each pid as ``CPU`` / ``GPU`` / ``Spans`` (and other torch.profiler
  string-pid tracks) using the per-rank ``process_labels`` metadata events.
* Always drops the ``Spans`` track (low-signal in this codebase, large in
  visual clutter).
* Optionally filters to just ``cpu`` or ``gpu`` events via ``--include``.
* Sorts the surviving tracks into contiguous Perfetto sections:
  **all CPU tracks (rank 0..N) first, then all GPU tracks (rank 0..N, stream
  0..K)**.
* Remaps every event's ``pid`` and flow ``id`` so cross-rank events never
  collide on the same track or flow arrow.

Because torch.profiler emits ``baseTimeNanoseconds`` from the same node clock,
timestamps line up directly across ranks — no time-shift needed for single-node
runs (multi-node would need clock-skew correction, not implemented here).

Examples
--------
Stitch step 52, default (CPU + GPU, drop Spans), gzip output::

    python scripts/stitch_traces.py <trace_dir> --step 52 --gzip

GPU-only view (skip CPU thread tree entirely — useful for kernel-level analysis)::

    python scripts/stitch_traces.py <trace_dir> --step 52 --include gpu --gzip

CPU-only view (host-side ops, profiler annotations, comm scheduling)::

    python scripts/stitch_traces.py <trace_dir> --step 52 --include cpu --gzip
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# trace_step52_rank3.json or trace_3_rank0.json (legacy filename)
_RANK_RE = re.compile(r"trace_(?:step)?(\d+)_rank(\d+)\.json$")
_KEY_RE = re.compile(r"trace_(.+?)_rank\d+\.json$")

# Per-rank pid offset. Picked large enough that no real OS pid collides
# (Linux pids fit in 22 bits; 1e6 per rank gives ~10 ranks of headroom).
_PID_STRIDE = 1_000_000

# Per-rank flow-id offset. torch.profiler flow ids are int32/int64 — pack rank
# into the high bits so cross-rank flows can never link by accident.
_FLOW_ID_STRIDE = 1 << 40

# Sort-index sections in Perfetto. Lower = appears higher in the timeline UI.
# Each section reserves a wide range so within-section ordering (rank, stream)
# fits comfortably without overlapping the next section.
_SORT_BASE = {
    "cpu":   0,
    "gpu":   1_000_000,
    "other": 10_000_000,   # Traces / "" misc string-pid tracks
}

# `Spans` carries no useful content in our workloads (one X event per trace)
# and clutters the timeline — always dropped.
_ALWAYS_DROP_PIDS_STR = {"Spans"}


def _classify_pid(pid_to_label: dict, pid_to_name: dict) -> dict:
    """Map original pid -> ('cpu'|'gpu'|'spans'|'other', stream_idx_or_0).

    Classification order, first match wins:
      1. pid (as a string) is in the always-drop set        -> 'spans'
      2. process_name is in the always-drop set             -> 'spans'
      3. process_labels == 'CPU'                            -> 'cpu'
      4. process_labels starts with 'GPU '                  -> 'gpu', stream id
      5. anything else (including unlabeled pids)           -> 'other'
    """
    all_pids = set(pid_to_label) | set(pid_to_name)
    out: dict = {}
    for pid in all_pids:
        label = pid_to_label.get(pid, "")
        name = pid_to_name.get(pid, "")
        if isinstance(pid, str) and pid in _ALWAYS_DROP_PIDS_STR:
            out[pid] = ("spans", 0)
            continue
        if name in _ALWAYS_DROP_PIDS_STR:
            out[pid] = ("spans", 0)
            continue
        if label == "CPU":
            out[pid] = ("cpu", 0)
        elif label.startswith("GPU"):
            try:
                stream_idx = int(label.split()[1])
            except (IndexError, ValueError):
                stream_idx = 0
            out[pid] = ("gpu", stream_idx)
        else:
            out[pid] = ("other", 0)
    return out


def _scan_pid_metadata(events: list[dict]) -> tuple[dict, dict]:
    """First pass: collect per-pid label and name from ``ph='M'`` events."""
    label: dict = {}
    name: dict = {}
    for e in events:
        if e.get("ph") != "M":
            continue
        pid = e.get("pid")
        if pid is None:
            continue
        if e.get("name") == "process_labels":
            label[pid] = e.get("args", {}).get("labels", "")
        elif e.get("name") == "process_name":
            name[pid] = e.get("args", {}).get("name", "")
    return label, name


def _new_sort_index(kind: str, rank: int, stream_idx: int) -> int:
    """Compute Perfetto sort_index so tracks group as: CPU(rank0..N), GPU(rank0..N, stream0..K), other."""
    base = _SORT_BASE.get(kind, _SORT_BASE["other"])
    return base + rank * 100 + stream_idx


def _new_pid(orig_pid, rank: int) -> object:
    """Remap a single pid into a per-rank namespace, preserving int vs str."""
    if isinstance(orig_pid, int):
        return orig_pid + rank * _PID_STRIDE
    if isinstance(orig_pid, str):
        try:
            return int(orig_pid) + rank * _PID_STRIDE
        except ValueError:
            return f"rank{rank}_{orig_pid}" if orig_pid else f"rank{rank}_misc"
    return orig_pid


def _process_one_rank(
    events: list[dict],
    rank: int,
    include: set[str],
) -> list[dict]:
    """Filter + remap one rank's events. ``include`` is a subset of {'cpu','gpu','other'}."""
    label, name = _scan_pid_metadata(events)
    classify = _classify_pid(label, name)

    out: list[dict] = []
    for e in events:
        pid = e.get("pid")
        if pid is None:
            out.append(e)
            continue
        # Always-drop check on the raw pid value first - Spans events in our
        # workloads have NO process_name/process_labels metadata, so the
        # classifier table doesn't list them. Catch them here directly.
        if isinstance(pid, str) and pid in _ALWAYS_DROP_PIDS_STR:
            continue
        kind, stream_idx = classify.get(pid, ("other", 0))
        if kind == "spans":          # always dropped
            continue
        if kind not in include:      # filtered by --include
            continue

        # Remap pid + flow id (per-rank namespace).
        e["pid"] = _new_pid(pid, rank)
        if "id" in e and e.get("ph") in ("s", "t", "f"):
            try:
                e["id"] = int(e["id"]) + rank * _FLOW_ID_STRIDE
            except (TypeError, ValueError):
                pass

        # Rewrite metadata: section-aware sort_index + rank-prefixed name.
        if e.get("ph") == "M":
            args = e.setdefault("args", {})
            if e.get("name") == "process_sort_index":
                args["sort_index"] = _new_sort_index(kind, rank, stream_idx)
            elif e.get("name") == "process_name":
                orig = args.get("name", "python")
                args["name"] = f"[Rank {rank}] {orig}"

        out.append(e)

    return out


def _group_by_step(trace_dir: Path) -> dict[str, dict[int, Path]]:
    """Map step-key (e.g. ``"step52"`` or ``"3"``) -> {rank: path}."""
    groups: dict[str, dict[int, Path]] = defaultdict(dict)
    for p in sorted(trace_dir.glob("trace_*_rank*.json")):
        m = _RANK_RE.search(p.name)
        if not m:
            continue
        prefix_match = _KEY_RE.match(p.name)
        key = prefix_match.group(1) if prefix_match else m.group(1)
        groups[key][int(m.group(2))] = p
    return dict(groups)


def stitch_one(rank_to_path: dict[int, Path], out_path: Path, *,
               include: set[str], gzip_out: bool, verbose: bool) -> None:
    """Merge one (step, rank->path) group into a single trace file."""
    merged_events: list[dict] = []
    base: dict | None = None

    for rank in sorted(rank_to_path):
        path = rank_to_path[rank]
        if verbose:
            sz_mb = path.stat().st_size / (1 << 20)
            print(f"  rank {rank}: {path.name} ({sz_mb:.1f} MB)", file=sys.stderr)
        with path.open() as f:
            trace = json.load(f)
        if base is None:
            base = {k: v for k, v in trace.items() if k != "traceEvents"}
            base["distributedInfo"] = {
                **trace.get("distributedInfo", {}),
                "stitched_ranks": sorted(rank_to_path),
                "stitched_files": [p.name for p in rank_to_path.values()],
                "stitched_include": sorted(include),
            }
        merged_events.extend(
            _process_one_rank(trace.get("traceEvents", []), rank, include)
        )

    assert base is not None, "no input traces provided"
    base["traceEvents"] = merged_events

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_out:
        with gzip.open(out_path, "wt") as f:
            json.dump(base, f)
    else:
        with out_path.open("w") as f:
            json.dump(base, f)
    if verbose:
        sz_mb = out_path.stat().st_size / (1 << 20)
        print(
            f"  -> {out_path}  ({len(merged_events):,} events, {sz_mb:.1f} MB)",
            file=sys.stderr,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("trace_dir", type=Path,
                    help="Directory containing trace_*_rank*.json files.")
    ap.add_argument("--step", type=str, default=None,
                    help="Stitch only the given step key (e.g. '52' or 'step52'). "
                         "Default: stitch every step group found.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path. Only valid when --step selects exactly "
                         "one group. Default: <trace_dir>/trace_<step>.json[.gz] "
                         "(or trace_<step>_cpu/_gpu when --include filters).")
    ap.add_argument("--include", choices=("cpu", "gpu", "both"), default="both",
                    help="Which sections to keep: cpu-only tracks, gpu-only "
                         "tracks, or both (default). 'Spans' is always dropped.")
    ap.add_argument("--gzip", action="store_true",
                    help="Write gzip-compressed JSON (Perfetto auto-detects).")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    if not args.trace_dir.is_dir():
        print(f"error: {args.trace_dir} is not a directory", file=sys.stderr)
        return 2

    if args.include == "both":
        # 'other' covers torch.profiler string-pid tracks (Traces / misc) that
        # carry low-volume but legitimate annotations. Dropped under cpu/gpu
        # so each filtered view is clean.
        include = {"cpu", "gpu", "other"}
    else:
        include = {args.include}

    groups = _group_by_step(args.trace_dir)
    if not groups:
        print(f"error: no trace_*_rank*.json files under {args.trace_dir}",
              file=sys.stderr)
        return 2

    if args.step is not None:
        wanted = args.step if args.step.startswith("step") else f"step{args.step}"
        if wanted not in groups and args.step in groups:
            wanted = args.step
        if wanted not in groups:
            print(
                f"error: step {args.step!r} not found. "
                f"Available: {sorted(groups)}",
                file=sys.stderr,
            )
            return 2
        groups = {wanted: groups[wanted]}

    if args.out is not None and len(groups) != 1:
        print("error: --out requires --step to select exactly one group",
              file=sys.stderr)
        return 2

    for key, rank_map in sorted(groups.items()):
        if not args.quiet:
            print(
                f"stitching {key} ({len(rank_map)} ranks, include={args.include}):",
                file=sys.stderr,
            )
        if args.out is not None:
            out = args.out
        else:
            ext = ".json.gz" if args.gzip else ".json"
            # Default mode ("both") gets the bare filename; explicit cpu/gpu
            # filters tag the output so they can coexist in one directory.
            suffix = "" if args.include == "both" else f"_{args.include}"
            out = args.trace_dir / f"trace_{key}{suffix}{ext}"
        stitch_one(rank_map, out, include=include,
                   gzip_out=args.gzip, verbose=not args.quiet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
