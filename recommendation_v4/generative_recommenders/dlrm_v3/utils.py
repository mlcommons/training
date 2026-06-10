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

# pyre-unsafe
"""
mlperf dlrm_v3 inference benchmarking tool.
"""

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gin
import tensorboard  # @manual=//tensorboard:lib  # noqa: F401 - required implicit dep when using torch.utils.tensorboard
import torch
from generative_recommenders.dlrm_v3.datasets.dataset import DLRMv3RandomDataset
from generative_recommenders.dlrm_v3.datasets.kuairand import DLRMv3KuaiRandDataset
from generative_recommenders.dlrm_v3.datasets.movie_lens import DLRMv3MovieLensDataset
from generative_recommenders.dlrm_v3.datasets.synthetic_movie_lens import (
    DLRMv3SyntheticMovieLensDataset,
)
from generative_recommenders.dlrm_v3.datasets.synthetic_streaming import (
    DLRMv3SyntheticStreamingDataset,
)
from generative_recommenders.dlrm_v3.datasets.yambda import DLRMv3YambdaDataset
from generative_recommenders.modules.multitask_module import (
    MultitaskTaskType,
    TaskConfig,
)
from torch.profiler import profile, profiler, ProfilerActivity  # pyre-ignore [21]
from torch.utils.tensorboard import SummaryWriter
from torchrec.metrics.accuracy import AccuracyMetricComputation
from torchrec.metrics.auc import AUCMetricComputation, compute_auc
from torchrec.metrics.gauc import GAUCMetricComputation
from torchrec.metrics.mae import MAEMetricComputation
from torchrec.metrics.metrics_namespace import MetricName, MetricPrefix
from torchrec.metrics.mse import MSEMetricComputation
from torchrec.metrics.ne import NEMetricComputation
from torchrec.metrics.rec_metric import (
    MetricComputationReport,
    RecMetricComputation,
)


class LifetimeAUCMetricComputation(AUCMetricComputation):
    """AUC over a 10M-sample (~5.5 eval-window) trailing buffer; emits with the
    LIFETIME prefix.

    NOTE: despite the name, this is NOT an uncapped since-step-0 AUC. The parent
    ``AUCMetricComputation`` evicts the prediction/label/weight buffers down to
    ``window_size`` in ``update()``; we instantiate it with
    ``window_size=10_000_000``, so "lifetime" is a ~10M-sample trailing window.
    Raise ``window_size`` (accepting unbounded buffer growth) if true cumulative
    AUC is ever required.

    Checkpoint correctness: torchrec registers the PREDICTIONS/LABELS/WEIGHTS
    buffers with ``persistent=False`` (so the default ``state_dict()`` drops
    them) and tracks a separate ``self._num_samples`` counter. Without the
    overrides below, every checkpoint resume would silently restart this metric
    from an empty buffer. We therefore serialize the buffers AND ``_num_samples``
    explicitly; restoring ``_num_samples`` is mandatory, since leaving it at 0
    makes the next ``update()`` take the init-sentinel branch and desync the
    windowed eviction. These buffers are per-rank-local (cross-rank gather only
    happens transiently at compute time), so the checkpoint layer MUST persist
    and restore them per-rank — see ``checkpoint.py``.
    """

    # Prefix used for the explicitly-serialized non-persistent buffers so the
    # keys can't collide with any persistent state the parent might register.
    _LIFETIME_KEY_PREFIX: str = "_lifetime_"

    def _compute(self) -> List[MetricComputationReport]:
        from typing import cast as _cast
        from torchrec.metrics.auc import LABELS, PREDICTIONS, WEIGHTS
        return [
            MetricComputationReport(
                name=MetricName.AUC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=compute_auc(
                    self._n_tasks,
                    _cast(List[torch.Tensor], getattr(self, PREDICTIONS)),
                    _cast(List[torch.Tensor], getattr(self, LABELS)),
                    _cast(List[torch.Tensor], getattr(self, WEIGHTS)),
                    self._apply_bin,
                ),
            )
        ]

    def lifetime_sample_count(self) -> int:
        """Current number of buffered samples (greppable for sanity logs)."""
        return int(getattr(self, "_num_samples", 0))

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        from torchrec.metrics.auc import LABELS, PREDICTIONS, WEIGHTS

        destination = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        # The parent registers these buffers persistent=False, so they are absent
        # from `destination`. Concatenate each buffer list to one (n_tasks, N)
        # tensor and serialize it alongside the sample counter.
        for attr in (PREDICTIONS, LABELS, WEIGHTS):
            buf = getattr(self, attr)
            if isinstance(buf, (list, tuple)) and len(buf) > 0:
                flat = torch.cat([t for t in buf], dim=-1)
            elif isinstance(buf, torch.Tensor):
                flat = buf
            else:
                flat = torch.empty(0)
            destination[prefix + self._LIFETIME_KEY_PREFIX + attr] = (
                flat.detach().cpu().clone()
            )
        destination[prefix + self._LIFETIME_KEY_PREFIX + "num_samples"] = (
            torch.tensor(int(getattr(self, "_num_samples", 0)), dtype=torch.long)
        )
        return destination

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
    ) -> Any:
        from torchrec.metrics.auc import LABELS, PREDICTIONS, WEIGHTS

        # Copy so we can strip our custom keys before delegating to the parent
        # (whose strict load would otherwise reject them as unexpected).
        remaining = dict(state_dict)
        saved_bufs: Dict[str, torch.Tensor] = {}
        for attr in (PREDICTIONS, LABELS, WEIGHTS):
            key = self._LIFETIME_KEY_PREFIX + attr
            if key in remaining:
                saved_bufs[attr] = remaining.pop(key)
        num_key = self._LIFETIME_KEY_PREFIX + "num_samples"
        saved_num = remaining.pop(num_key, None)

        result = super().load_state_dict(remaining, strict=strict)

        if saved_bufs:
            # Device of the live (init-sentinel) buffers; keep restored buffers
            # co-located so subsequent update()/compute() stay on-device.
            existing = getattr(self, PREDICTIONS)
            dev = (
                existing[0].device
                if isinstance(existing, (list, tuple)) and len(existing) > 0
                else torch.device("cpu")
            )
            for attr, val in saved_bufs.items():
                setattr(self, attr, [val.to(dev)])
            if saved_num is not None:
                self._num_samples = int(saved_num.item())
        return result


# Sentinel "window size" used for the FRESH eval metrics so torchrec's windowed
# eviction never fires within a single eval pass (the per-pass reset bounds the
# buffer to exactly one full holdout pass). 1<<60 is far above any realistic
# per-rank sample count and avoids sys.maxsize overflow inside torchrec math.
UNBOUNDED_WINDOW: int = 1 << 60


class BinnedCumulativeAUC(RecMetricComputation):
    """Cumulative AUC via a fixed-resolution score histogram (LIFETIME prefix).

    Global AUC is a rank statistic, so it has no fixed-size additive sufficient
    statistic the way NE/Accuracy do - exact cumulative AUC otherwise needs every
    (score, label) pair retained and sorted (the buffer-based ``AUCMetricComputation``
    / ``LifetimeAUCMetricComputation``). Instead we keep two weighted histograms of
    positive/negative mass per score bin. This gives an AUC exact up to bin width
    with O(num_bins) memory that does NOT grow with sample count, and - because
    histograms are additive - cross-rank sync is a cheap all-reduce (dist_reduce_fx
    "sum") rather than all-gathering millions of predictions. The state is truly
    cumulative across all eval passes (never evicted, never reset on eval).

    Predictions MUST be probabilities in [0, 1] (the same tensor feeds NE, which
    requires probabilities; the model applies sigmoid in multitask_module). Values
    are clamped into [0, 1] defensively.
    """

    def __init__(self, *args, num_bins: int = 100_000, **kwargs) -> None:
        # window_size is irrelevant here (no windowed state); pass through.
        super().__init__(*args, **kwargs)
        self._num_bins: int = int(num_bins)
        self._add_state(
            "pos_hist",
            torch.zeros((self._n_tasks, self._num_bins), dtype=torch.float64),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=True,
        )
        self._add_state(
            "neg_hist",
            torch.zeros((self._n_tasks, self._num_bins), dtype=torch.float64),
            add_window_state=False,
            dist_reduce_fx="sum",
            persistent=True,
        )

    def cumulative_sample_count(self) -> int:
        """Total weighted samples in the histograms (greppable for sanity logs)."""
        return int((self.pos_hist.sum() + self.neg_hist.sum()).item())

    def update(
        self,
        *,
        predictions: Optional[torch.Tensor],
        labels: torch.Tensor,
        weights: Optional[torch.Tensor],
        **kwargs: Dict[str, Any],
    ) -> None:
        if predictions is None or weights is None:
            raise ValueError(
                "BinnedCumulativeAUC.update requires predictions and weights"
            )
        preds = predictions.float().clamp_(0.0, 1.0)  # (n_tasks, n_examples)
        labels = labels.float()
        weights = weights.float()
        # Bin index per example; the top edge (p==1.0) folds into the last bin.
        idx = (preds * self._num_bins).long().clamp_(0, self._num_bins - 1)
        pos_w = (weights * labels).to(self.pos_hist.dtype)
        neg_w = (weights * (1.0 - labels)).to(self.neg_hist.dtype)
        self.pos_hist.scatter_add_(1, idx, pos_w)
        self.neg_hist.scatter_add_(1, idx, neg_w)

    def _compute(self) -> List[MetricComputationReport]:
        # By compute() time torchmetrics has all-reduced (summed) the histograms
        # across ranks, so these are the global per-bin masses.
        pos = self.pos_hist  # (n_tasks, num_bins)
        neg = self.neg_hist
        total_pos = pos.sum(dim=1)
        total_neg = neg.sum(dim=1)
        # Lower bin index == lower score. A positive in bin b outranks every
        # negative in bins < b (exclusive prefix sum), and ties in bin b score
        # 0.5. AUC = sum_b pos_b * (neg_below_b + 0.5*neg_b) / (P * N).
        neg_below = torch.cumsum(neg, dim=1) - neg
        numerator = (pos * (neg_below + 0.5 * neg)).sum(dim=1)
        denom = total_pos * total_neg
        auc = torch.where(
            denom > 0,
            numerator / denom,
            torch.full_like(numerator, 0.5),
        ).to(torch.float32)
        return [
            MetricComputationReport(
                name=MetricName.AUC,
                metric_prefix=MetricPrefix.LIFETIME,
                value=auc,
            )
        ]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("utils")


def _trim_warmup_from_trace(path: str, keep_n_active: int) -> None:
    """Post-process a chrome trace to drop events from WARMUP-phase steps.

    torch.profiler captures events during BOTH the WARMUP and RECORD phases
    of a schedule and writes them all to the exported trace. There is no
    built-in flag to exclude WARMUP from the export. We approximate it by:

      1) Finding all ``ProfilerStep#N`` spans in the file.
      2) Keeping only the last ``keep_n_active`` of them (sorted by start
         timestamp) as the "active" range.
      3) Filtering ``traceEvents`` to events whose ``ts`` falls inside that
         range. Metadata events (``ph='M'``) are always preserved.

    Mutates the file in place.
    """
    import json as _json
    with open(path) as f:
        d = _json.load(f)
    events = d.get("traceEvents", [])

    # ProfilerStep spans mark training-step boundaries; we filter by their
    # time ranges rather than by name index because step numbering can offset
    # between schedule_fn argument and the value printed in the trace.
    # torch.profiler emits one ProfilerStep#N span per CPU thread that ran
    # during that step, so dedupe by name first so "5 active steps" means
    # 5 distinct step numbers, not 5 spans.
    name_to_span: Dict[str, tuple] = {}
    for e in events:
        nm = e.get("name", "")
        if "ProfilerStep" not in nm or e.get("ph") != "X" or "ts" not in e:
            continue
        ts = e["ts"]
        end = ts + e.get("dur", 0)
        prev = name_to_span.get(nm)
        if prev is None:
            name_to_span[nm] = (ts, end)
        else:
            name_to_span[nm] = (min(prev[0], ts), max(prev[1], end))
    if len(name_to_span) <= keep_n_active:
        return
    sorted_spans = sorted(name_to_span.values())
    active = sorted_spans[-keep_n_active:]
    t_start = min(s for s, _ in active)
    t_end = max(e for _, e in active)

    def _keep(e: dict) -> bool:
        if e.get("ph") == "M":
            return True
        ts = e.get("ts")
        if ts is None:
            return True
        return t_start <= ts < t_end

    kept = [e for e in events if _keep(e)]
    d["traceEvents"] = kept
    with open(path, "w") as f:
        _json.dump(d, f)
    logger.warning(
        f"Trimmed WARMUP events from {path}: {len(events):,} -> {len(kept):,} "
        f"(kept active range [{t_start:.0f}, {t_end:.0f}] us)"
    )


# GPU activity categories used to detect GPU stream rows and their busy time.
_GPU_KERNEL_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})


def _is_rocm() -> bool:
    """True on ROCm/AMD builds (``torch.version.hip`` set), False on CUDA/B200.

    The ProfilerStep-layout normalization and the sub-us kernel de-overlap are
    workarounds for how roctracer projects annotations/kernels onto HIP streams;
    CUDA/CUPTI traces don't have those artifacts, so these passes must be skipped
    on NVIDIA to avoid touching otherwise-correct traces.
    """
    return getattr(torch.version, "hip", None) is not None


def _normalize_profilerstep_layout(path: str) -> None:
    """Collapse fragmented GPU-side ``ProfilerStep#N`` spans into one span/step.

    ``torch.profiler`` emits ``ProfilerStep#N`` as a CPU ``user_annotation`` that
    Kineto projects onto the GPU timeline as ``gpu_user_annotation`` spans. On
    CUDA the blocking H2D copy shares the compute stream, so each step projects
    onto a single GPU stream and renders as one full-width span. On ROCm a
    blocking H2D copy lands on HIP's null stream (a different stream than the
    non-null compute stream), so the step splits across two GPU rows and looks
    truncated in Perfetto — a pure rendering artifact (every kernel is still
    captured, and the underlying GPU is busy for the whole step).

    This rewrites each per-step GPU ``ProfilerStep`` annotation to a single span
    on the rank's busiest (compute) GPU stream, covering the kernel extent inside
    that step's CPU window. Works on a raw per-rank trace (GPU streams are tids
    under one pid) by keying the busiest stream on ``(pid, tid)``. No-op when the
    annotation already lives on a single GPU stream (the CUDA case), so it is
    safe to run on every platform. Mutates the file in place.
    """
    import json as _json

    with open(path) as f:
        d = _json.load(f)
    events = d.get("traceEvents", [])

    # Per (pid,tid) GPU busy time -> identify the busiest = compute stream.
    stream_busy: Dict[tuple, int] = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in _GPU_KERNEL_CATS:
            dur = e.get("dur", 0)
            if dur > 0:
                key = (e.get("pid"), e.get("tid"))
                stream_busy[key] = stream_busy.get(key, 0) + dur
    if not stream_busy:
        return
    busiest = max(stream_busy, key=lambda k: stream_busy[k])

    # Existing GPU-side ProfilerStep spans and the streams they sit on.
    gpu_ps_streams = set()
    template = None
    for e in events:
        if e.get("cat") == "gpu_user_annotation" and str(
            e.get("name", "")
        ).startswith("ProfilerStep"):
            gpu_ps_streams.add((e.get("pid"), e.get("tid")))
            if template is None:
                template = e
    # No fragmentation (single stream or none) -> leave the trace untouched.
    if len(gpu_ps_streams) <= 1:
        return

    # CPU ProfilerStep windows: step name -> [min ts, max end].
    cpu_win: Dict[str, list] = {}
    for e in events:
        if (
            e.get("cat") == "user_annotation"
            and e.get("ph") == "X"
            and str(e.get("name", "")).startswith("ProfilerStep")
        ):
            ts = e.get("ts", 0)
            end = ts + e.get("dur", 0)
            w = cpu_win.get(e["name"])
            if w is None:
                cpu_win[e["name"]] = [ts, end]
            else:
                w[0] = min(w[0], ts)
                w[1] = max(w[1], end)

    # GPU kernel extents (any stream) for clamping each step's span.
    gpu_kernels = [
        (e.get("ts", 0), e.get("ts", 0) + e.get("dur", 0))
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") in _GPU_KERNEL_CATS
        and e.get("dur", 0) > 0
    ]

    new_spans = []
    for sname, (cs, ce) in cpu_win.items():
        ks = [(ts, end) for ts, end in gpu_kernels if end > cs and ts < ce]
        if not ks:
            continue
        gmin = min(ts for ts, _ in ks)
        gmax = max(end for _, end in ks)
        span = dict(template) if template else {}
        span.update(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": sname,
                "pid": busiest[0],
                "tid": busiest[1],
                "ts": gmin,
                "dur": gmax - gmin,
                "args": {"normalized_profilerstep": True},
            }
        )
        new_spans.append(span)

    if not new_spans:
        return

    out = [
        e
        for e in events
        if not (
            e.get("cat") == "gpu_user_annotation"
            and str(e.get("name", "")).startswith("ProfilerStep")
        )
    ]
    dropped = len(events) - len(out)
    out.extend(new_spans)
    d["traceEvents"] = out
    with open(path, "w") as f:
        _json.dump(d, f)
    logger.warning(
        f"Normalized GPU ProfilerStep layout in {path}: dropped {dropped} "
        f"fragmented span(s) across {len(gpu_ps_streams)} stream(s), wrote "
        f"{len(new_spans)} span(s) on busiest stream pid={busiest[0]} "
        f"tid={busiest[1]}"
    )


def _deoverlap_gpu_slices(path: str, max_snap_us: float = 5.0) -> None:
    """Remove sub-microsecond kernel overlaps that break Perfetto's renderer.

    Perfetto draws all ``ph=="X"`` slices on a single track (one ``(pid, tid)``)
    as a strict nested stack ordered by start time: a slice that *opens* while a
    previous slice on the same track is still open is treated as that slice's
    child and is **clipped to the parent's end**. ROCm's roctracer reports
    per-stream kernel timestamps at ns granularity, so two back-to-back kernels
    on the same compute stream occasionally overlap by a fraction of a
    microsecond (e.g. an 88 ns ``elementwise`` epilogue ending 0.075 us *after*
    the next 21 ms ``_hstu_attn_bwd`` kernel begins). Perfetto then nests the
    long kernel inside the tiny one and clips it to a sub-pixel sliver, so the
    kernel "disappears" from the timeline even though it is fully present in the
    JSON.

    This pulls each slice's end back to just *before* the next slice's start
    whenever they overlap by less than ``max_snap_us`` (a measurement artifact,
    not real concurrency — kernels on one stream are serialized), leaving genuine
    nesting (a small kernel fully contained in a larger one) untouched. The
    adjustment is sub-microsecond and does not change any reported duration
    meaningfully. Mutates the file in place; best-effort.

    Critically, the slices are separated by a tiny ``_GAP_US`` (~1 ns) rather
    than snapped to an *exactly equal* end==start timestamp. A coincident
    end==start is just as fatal as an overlap in Perfetto: it nests the next
    slice inside the previous one and clips it to zero width (this is the ~1 ns
    gap that roctracer leaves between cleanly-rendered back-to-back kernels). So
    we also fix exact-touch (``a_end == b.ts``) boundaries, not just overlaps.
    """
    import json as _json
    from collections import defaultdict

    # ~1 ns. Matches the natural inter-kernel gap roctracer leaves between
    # back-to-back kernels that Perfetto already renders correctly. Must be
    # strictly > 0 so end != start after the nudge.
    _GAP_US = 0.001

    with open(path) as f:
        d = _json.load(f)
    events = d.get("traceEvents", [])

    tracks: Dict[tuple, list] = defaultdict(list)
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("cat") in _GPU_KERNEL_CATS
            and e.get("dur", 0) > 0
        ):
            tracks[(e.get("pid"), e.get("tid"))].append(e)

    snapped = 0
    max_clip = 0.0
    for sl in tracks.values():
        # Sort by start, then longest-first so a container precedes the slices
        # it nests; consecutive pairs are then either disjoint, properly nested,
        # or a tiny artifact overlap.
        sl.sort(key=lambda e: (e["ts"], -e["dur"]))
        for i in range(len(sl) - 1):
            a = sl[i]
            b = sl[i + 1]
            a_end = a["ts"] + a["dur"]
            b_end = b["ts"] + b["dur"]
            # Touching (a_end == b.ts) or partial overlap (a ends inside b) both
            # break rendering; true containment (a_end >= b_end) is valid nesting
            # and is left alone.
            if b["ts"] <= a_end < b_end:
                desired_end = b["ts"] - _GAP_US
                clip = a_end - desired_end
                if a["ts"] < desired_end and 0 < clip < max_snap_us:
                    a["dur"] = desired_end - a["ts"]
                    snapped += 1
                    if clip > max_clip:
                        max_clip = clip

    if snapped:
        with open(path, "w") as f:
            _json.dump(d, f)
        logger.warning(
            f"De-overlapped GPU slices in {path}: snapped {snapped} sub-us "
            f"overlap(s) (max {max_clip:.3f}us) so Perfetto renders every kernel"
        )


def _deoverlap_gpu_annotations(path: str, max_snap_us: float = 5.0) -> None:
    """Separate touching/overlapping *sibling* GPU annotations so Perfetto draws
    each one full width (the B200-style stacked layout).

    Same root cause as :func:`_deoverlap_gpu_slices`, but at the annotation
    boundary instead of the kernel boundary. The forward/backward phase
    annotations Kineto projects onto the GPU stream (``## item_forward ##``,
    ``## user_forward ##``, ``## multitask_module ##``, the ``## stu_* ##``
    pairs, ...) are emitted as a chain of siblings laid end-to-end: each is meant
    to end exactly where the next begins. Perfetto stores timestamps as int64 ns,
    and the absolute step timestamps are ~5.4e12 us where a float64's quantum is
    already ~1 ns, so a sibling boundary that should be coincident instead lands
    a few ns off. When the earlier sibling's end falls *at or after* the next
    sibling's start, Perfetto nests the next sibling inside it and clips it to a
    sub-pixel sliver — so e.g. the 100+ ms ``## user_forward ##`` span vanishes on
    some ranks/steps and renders on others purely by rounding luck.

    Unlike kernels (all flat on one stream), annotations form a real nesting
    hierarchy — ``## user_forward ##`` legitimately *contains* the ``## stu_* ##``
    spans and their kernels — so this cannot blindly snap consecutive slices. It
    walks the per-track slice stack (sorted by start, longest-first) and only
    snaps a slice ``a`` back when the next slice ``b`` is **not** contained in it
    (``b`` extends beyond ``a``'s end), i.e. they are siblings rather than
    parent/child. Real containment is left untouched, and a snap is skipped if it
    would clip into ``a``'s own descendants (kernels or child annotations).
    Mutates the file in place; best-effort. Run after :func:`_deoverlap_gpu_slices`
    so kernel boundaries are already clean.
    """
    import json as _json
    from collections import defaultdict

    # ~2 ns. The annotation boundaries sit at ~5.4e12 us where a float64's
    # quantum is ~0.98 ns, so a 1 ns nudge can round back onto the neighbour's
    # timestamp (an exact touch, which Perfetto still nests+clips). 2 ns (~2
    # quanta) reliably separates them and is still far below any visible width.
    _GAP_US = 0.002

    with open(path) as f:
        d = _json.load(f)
    events = d.get("traceEvents", [])

    # Stack the full per-track hierarchy over BOTH kernels and annotations so a
    # parent annotation knows the extent of its descendants (the snap guard),
    # but only annotation slices are ever trimmed.
    _ANN = "gpu_user_annotation"
    tracks: Dict[tuple, list] = defaultdict(list)
    for e in events:
        if (
            e.get("ph") == "X"
            and e.get("dur", 0) > 0
            and (e.get("cat") in _GPU_KERNEL_CATS or e.get("cat") == _ANN)
        ):
            tracks[(e.get("pid"), e.get("tid"))].append(e)

    snapped = 0
    max_clip = 0.0
    for sl in tracks.values():
        # Longest-first on ties so a container precedes the slices it nests.
        sl.sort(key=lambda e: (e["ts"], -e["dur"]))
        # Each frame: [event, max_descendant_end]. The stack holds the chain of
        # currently-open ancestors for the slice being placed.
        stack: list = []
        for b in sl:
            b_ts = b["ts"]
            b_end = b_ts + b["dur"]
            while stack:
                a = stack[-1][0]
                a_end = a["ts"] + a["dur"]
                if a_end < b_ts:
                    # a closed strictly before b begins -> disjoint sibling, pop.
                    frame = stack.pop()
                    eff = frame[0]["ts"] + frame[0]["dur"]
                    if stack:
                        stack[-1][1] = max(stack[-1][1], eff, frame[1])
                    continue
                if a_end < b_end:
                    # b starts at/inside a but extends past a's end => they are
                    # siblings (not parent/child), and a's tail nests+clips b in
                    # Perfetto. Snap a's end to just before b. This fires for both
                    # annotation tails (## item_forward ## overhanging
                    # ## user_forward ##) and kernel tails that straddle an
                    # annotation boundary (a layer-norm kernel ending a few ns
                    # past the start of the next phase span) -- both are sub-us
                    # roctracer/rounding artifacts, since kernels on one stream
                    # are serialized and phase spans are sequential.
                    desired_end = b_ts - _GAP_US
                    clip = a_end - desired_end
                    # Guard: only snap when a's deepest descendant ends at or
                    # before b's start. If a child (kernel or nested span)
                    # actually extends *past* b.ts, trimming a wouldn't fix b's
                    # clipping (the child would still nest b) and could drop a
                    # real child into b's territory, so leave it. A descendant
                    # ending exactly at the boundary is itself rounding noise and
                    # is clipped by <=1 ns, which is fine.
                    if (
                        a["ts"] < desired_end
                        and stack[-1][1] <= b_ts
                        and 0 < clip < max_snap_us
                    ):
                        a["dur"] = desired_end - a["ts"]
                        snapped += 1
                        if clip > max_clip:
                            max_clip = clip
                    frame = stack.pop()
                    eff = frame[0]["ts"] + frame[0]["dur"]
                    if stack:
                        stack[-1][1] = max(stack[-1][1], eff, frame[1])
                    continue
                # a_end >= b_end: a fully contains b -> b is a child, stop.
                break
            stack.append([b, b_ts])

    if snapped:
        with open(path, "w") as f:
            _json.dump(d, f)
        logger.warning(
            f"De-overlapped GPU annotations in {path}: snapped {snapped} sub-us "
            f"sibling overlap(s) (max {max_clip:.3f}us) so Perfetto renders every "
            f"annotation full width"
        )


def _on_trace_ready_fn(
    rank: Optional[int] = None,
    trace_dir: str = "/tmp/dlrm_v3_traces",
    keep_n_active: Optional[int] = None,
    trace_steps: Optional[List[int]] = None,
) -> Callable[[torch.profiler.profile], None]:
    """Create the on_trace_ready callback that exports a chrome trace to disk.

    Filename follows ``trace_step{step}_rank{rank}.json`` so multi-rank
    captures don't collide and ``scripts/stitch_traces.py`` can merge them
    by step number.

    The ``{step}`` label:

    * If ``trace_steps`` is provided (multi-window mode), the Nth callback
      invocation labels its file with ``trace_steps[N]`` -- i.e. the
      user-requested step that triggered the window. This is the most
      intuitive labelling.
    * Otherwise falls back to ``p.step_num`` (torch.profiler's internal
      counter at trigger time, off by ~warmup+active from the schedule
      arg).

    If ``keep_n_active`` is set, the exported file is post-processed to keep
    only the last N ProfilerStep-spans worth of events (i.e. drop WARMUP).
    """
    state = {"fire_count": 0}

    def handle_fn(p: torch.profiler.profile) -> None:
        os.makedirs(trace_dir, exist_ok=True)
        if trace_steps:
            i = state["fire_count"]
            step_label = (
                trace_steps[i] if i < len(trace_steps) else getattr(p, "step_num", 0)
            )
        else:
            step_label = getattr(p, "step_num", 0)
        state["fire_count"] += 1
        rank_str = f"_rank{rank}" if rank is not None else ""
        file_name = f"trace_step{step_label}{rank_str}.json"
        path = os.path.join(trace_dir, file_name)
        logger.warning(
            p.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total"
            )
        )
        # Tracing is best-effort: a write/trim failure (permissions, disk full,
        # malformed export) must never crash the training run. Degrade to a
        # warning so the loop continues — especially important since streaming
        # enables output_trace by default.
        try:
            p.export_chrome_trace(path)
            logger.warning(f"Trace written to: {path}")
            if keep_n_active is not None and keep_n_active > 0:
                _trim_warmup_from_trace(path, keep_n_active)
            # ROCm/AMD-only rendering fixes. CUDA/CUPTI (e.g. B200) traces don't
            # exhibit the fragmented-ProfilerStep or sub-us kernel-overlap
            # artifacts, so skip entirely on NVIDIA to avoid touching otherwise
            # correct traces. Best-effort like trim above.
            if _is_rocm():
                # Normalize the GPU-side ProfilerStep layout so ROCm traces
                # render with one full-width step span per stream like CUDA.
                _normalize_profilerstep_layout(path)
                # Snap roctracer's sub-us kernel overlaps so Perfetto doesn't
                # mis-nest and hide long kernels.
                _deoverlap_gpu_slices(path)
                # Same fix at the annotation-sibling boundary so phase spans
                # (## user_forward ##, ## stu_* ##, ...) render full width.
                _deoverlap_gpu_annotations(path)
        except Exception as exc:
            logger.warning(f"Trace export/trim failed for {path}: {exc!r} (skipping)")

    return handle_fn


def profiler_or_nullcontext(
    enabled: bool, with_stack: bool, trace_dir: str = "/tmp/dlrm_v3_traces"
):
    """One-shot profile context for ad-hoc captures (no scheduling)."""
    return (
        profile(
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=_on_trace_ready_fn(trace_dir=trace_dir),
            with_stack=with_stack,
        )
        if enabled
        else contextlib.nullcontext()
    )


def _multi_window_schedule(
    trace_steps,
    warmup: int,
    active: int,
):
    """Custom schedule that profiles around each step in ``trace_steps``.

    Step s gets:
        [s - warmup, s)        -> WARMUP
        [s, s + active - 1)    -> RECORD
        s + active - 1         -> RECORD_AND_SAVE
    """
    windows = [(s - warmup, s, s + active) for s in sorted(trace_steps)]

    def schedule_fn(step: int) -> torch.profiler.ProfilerAction:
        for warmup_start, active_start, active_end in windows:
            if warmup_start <= step < active_start:
                return torch.profiler.ProfilerAction.WARMUP
            if active_start <= step < active_end - 1:
                return torch.profiler.ProfilerAction.RECORD
            if step == active_end - 1:
                return torch.profiler.ProfilerAction.RECORD_AND_SAVE
        return torch.profiler.ProfilerAction.NONE

    return schedule_fn


@gin.configurable
class Profiler:
    """Scheduled torch.profiler wrapper that writes Chrome traces to disk.

    Two modes (set via gin):

    * Single window (default): ``wait=10, warmup=20, active=50, repeat=1``.
      Captures one contiguous window starting after ``wait`` steps.
    * Multi-window: ``trace_steps=[500, 1000, 5000]`` (overrides wait+repeat).
      Captures a separate window around each listed step.

    All knobs are gin-tunable, e.g. in a gin file::

        Profiler.trace_dir = "/path/to/results/exp42/trace"
        Profiler.trace_steps = [500, 1000, 5000]
        Profiler.warmup = 5
        Profiler.active = 10
    """

    def __init__(
        self,
        rank: int,
        active: int = 50,
        wait: int = 10,
        warmup: int = 20,
        repeat: int = 1,
        trace_steps: Optional[List[int]] = None,
        trace_dir: str = "/tmp/dlrm_v3_traces",
        trim_warmup: bool = True,
        record_shapes: bool = True,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
    ) -> None:
        self.rank = rank
        self.trace_dir = trace_dir
        if trace_steps:
            sched = _multi_window_schedule(trace_steps, warmup, active)
        else:
            sched = torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat
            )
        keep_n = active if trim_warmup else None
        self._profiler: profiler.profile = torch.profiler.profile(
            schedule=sched,
            on_trace_ready=_on_trace_ready_fn(
                self.rank, trace_dir, keep_n, trace_steps
            ),
            # pyre-fixme[16]: Module `profiler` has no attribute `ProfilerActivity`.
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
        )

    def step(self) -> None:
        """Advance the profiler to the next step."""
        self._profiler.step()


@gin.configurable
class MetricsLogger:
    """
    Logger for tracking and computing recommendation metrics.

    Supports both classification metrics (NE, Accuracy, GAUC) and regression
    metrics (MSE, MAE) based on multitask configuration.

    Args:
        multitask_configs: List of task configurations defining metric types.
        batch_size: Batch size for metric computation.
        window_size: Window size for running metric aggregation.
        device: Device to place metric tensors on.
        rank: Process rank for distributed training.
        tensorboard_log_path: Optional path for TensorBoard logging.
    """

    def __init__(
        self,
        multitask_configs: List[TaskConfig],
        batch_size: int,
        window_size: int,
        device: torch.device,
        rank: int,
        tensorboard_log_path: str = "",
        world_size: int = 1,
        auc_threshold: Optional[float] = None,
        num_flops_per_sample: float = 0.0,
        gpu_peak_flops: float = 0.0,
        model: Optional[torch.nn.Module] = None,
        eval_cumulative: bool = False,
        cumulative_auc_bins: int = 100_000,
        train_lifetime_auc_mode: str = "binned",
        eval_lifetime_auc_mode: str = "binned",
        lifetime_auc_window: int = 10_000_000,
    ) -> None:
        # tflops/mfu reporting state (optional — when both num_flops_per_sample
        # and gpu_peak_flops are set, the train perf line gains tflops_algo/gpu,
        # mfu, tflops_real/gpu, hfu, fill. The jagged ("real") numbers come
        # from `model._last_jagged_flops_per_sample` stashed by DlrmHSTU.main_forward.
        self._num_flops_per_sample: float = max(0.0, float(num_flops_per_sample))
        self._gpu_peak_flops: float = max(0.0, float(gpu_peak_flops))
        self._model_ref: Optional[torch.nn.Module] = model
        if rank == 0 and self._num_flops_per_sample > 0 and self._gpu_peak_flops > 0:
            logger.info(
                f"FLOPS reporting enabled: {self._num_flops_per_sample / 1e9:.1f} "
                f"GFLOP/sample (dense fwd+bwd), GPU peak {self._gpu_peak_flops / 1e12:.0f} TFLOPS"
            )
        self.multitask_configs: List[TaskConfig] = multitask_configs
        all_classification_tasks: List[str] = [
            task.task_name
            for task in self.multitask_configs
            if task.task_type != MultitaskTaskType.REGRESSION
        ]
        all_regression_tasks: List[str] = [
            task.task_name
            for task in self.multitask_configs
            if task.task_type == MultitaskTaskType.REGRESSION
        ]
        assert all_classification_tasks + all_regression_tasks == [
            task.task_name for task in multitask_configs
        ]
        self.task_names: List[str] = all_classification_tasks + all_regression_tasks

        # Eval metric semantics:
        #   eval_cumulative=False (default, legacy / static / non-streaming eval):
        #     a single eval set with the configured window_size, including a
        #     lifetime AUC. Unchanged behavior.
        #   eval_cumulative=True (streaming fixed-holdout eval): a FRESH eval set
        #     (window_size=UNBOUNDED, reset each pass -> per-pass full-holdout
        #     "window_*") PLUS a CUMULATIVE set ("eval_cum", never reset ->
        #     "lifetime_*"). NE/Accuracy/GAUC are cumulative for free via their
        #     persistent scalar sums; AUC cumulative uses the selected backend.
        #
        # Lifetime-AUC backend is configurable independently for train and eval:
        #   "binned" (default): BinnedCumulativeAUC - exact-cumulative AUC via an
        #     O(num_bins) score histogram (additive all-reduce, no unbounded
        #     buffer, memory independent of #samples/#windows).
        #   "capped": LifetimeAUCMetricComputation - AUC over a trailing buffer of
        #     `lifetime_auc_window` samples/rank (the legacy approach; per-rank
        #     buffer all-gathered at compute).
        self._eval_cumulative: bool = eval_cumulative
        self._cumulative_auc_bins: int = int(cumulative_auc_bins)
        self._train_lifetime_auc_mode: str = str(train_lifetime_auc_mode)
        self._eval_lifetime_auc_mode: str = str(eval_lifetime_auc_mode)
        self._lifetime_auc_window: int = int(lifetime_auc_window)
        n_cls = len(all_classification_tasks)
        n_reg = len(all_regression_tasks)

        def _make_lifetime_auc(mode: str) -> RecMetricComputation:
            if mode == "binned":
                # window_size=0: no torchrec windowed state; histograms only.
                return BinnedCumulativeAUC(
                    my_rank=rank, batch_size=batch_size, n_tasks=n_cls,
                    window_size=0, num_bins=self._cumulative_auc_bins,
                ).to(device)
            if mode == "capped":
                return LifetimeAUCMetricComputation(
                    my_rank=rank, batch_size=batch_size, n_tasks=n_cls,
                    window_size=self._lifetime_auc_window,
                ).to(device)
            raise ValueError(
                f"lifetime_auc_mode must be 'binned' or 'capped', got {mode!r}"
            )

        def _make_class(ws: int, lifetime_mode: Optional[str]) -> List[RecMetricComputation]:
            mets: List[RecMetricComputation] = [
                NEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
                AccuracyMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
                GAUCMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
                AUCMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=ws).to(device),
            ]
            if lifetime_mode is not None:
                mets.append(_make_lifetime_auc(lifetime_mode))
            return mets

        def _make_class_cumulative() -> List[RecMetricComputation]:
            # NE/Accuracy/GAUC: cumulative via persistent lifetime sums (window
            # value ignored at compute). AUC: selected lifetime backend.
            return [
                NEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=window_size).to(device),
                AccuracyMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=window_size).to(device),
                GAUCMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_cls, window_size=window_size).to(device),
                _make_lifetime_auc(self._eval_lifetime_auc_mode),
            ]

        def _make_reg(ws: int) -> List[RecMetricComputation]:
            return [
                MSEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_reg, window_size=ws).to(device),
                MAEMetricComputation(my_rank=rank, batch_size=batch_size, n_tasks=n_reg, window_size=ws).to(device),
            ]

        self.class_metrics: Dict[str, List[RecMetricComputation]] = {"train": [], "eval": []}
        self.regression_metrics: Dict[str, List[RecMetricComputation]] = {"train": [], "eval": []}
        if eval_cumulative:
            self.class_metrics["eval_cum"] = []
            self.regression_metrics["eval_cum"] = []

        if all_classification_tasks:
            self.class_metrics["train"] = _make_class(window_size, lifetime_mode=self._train_lifetime_auc_mode)
            if eval_cumulative:
                self.class_metrics["eval"] = _make_class(UNBOUNDED_WINDOW, lifetime_mode=None)
                self.class_metrics["eval_cum"] = _make_class_cumulative()
            else:
                self.class_metrics["eval"] = _make_class(window_size, lifetime_mode=self._eval_lifetime_auc_mode)

        if all_regression_tasks:
            self.regression_metrics["train"] = _make_reg(window_size)
            if eval_cumulative:
                self.regression_metrics["eval"] = _make_reg(UNBOUNDED_WINDOW)
                self.regression_metrics["eval_cum"] = _make_reg(window_size)
            else:
                self.regression_metrics["eval"] = _make_reg(window_size)

        self.global_step: Dict[str, int] = {"train": 0, "eval": 0}
        self.tb_logger: Optional[SummaryWriter] = None
        if tensorboard_log_path != "":
            self.tb_logger = SummaryWriter(log_dir=tensorboard_log_path, purge_step=0)
            self.tb_logger.flush()

        # Throughput / time-to-target tracking. Counters are train-only; eval
        # samples are not relevant for headline samples/sec numbers.
        self._world_size: int = max(1, int(world_size))
        self._auc_threshold: Optional[float] = auc_threshold
        self._time_to_target_logged: bool = False
        self._perf_t_start: float = time.perf_counter()
        self._perf_t_window: float = self._perf_t_start
        self._perf_steps_in_window: int = 0
        self._perf_total_samples: int = 0
        self._perf_samples_counter: torch.Tensor = torch.zeros(
            1, dtype=torch.long, device=device
        )

    @property
    def all_metrics(self) -> Dict[str, List[RecMetricComputation]]:
        """
        Get all metrics for train and eval modes.

        Returns:
            Dictionary mapping mode ('train'/'eval') to list of metric computations.
        """
        out = {
            "train": self.class_metrics["train"] + self.regression_metrics["train"],
            "eval": self.class_metrics["eval"] + self.regression_metrics["eval"],
        }
        if "eval_cum" in self.class_metrics or "eval_cum" in self.regression_metrics:
            out["eval_cum"] = self.class_metrics.get(
                "eval_cum", []
            ) + self.regression_metrics.get("eval_cum", [])
        return out

    def update(
        self,
        predictions: torch.Tensor,
        weights: torch.Tensor,
        labels: torch.Tensor,
        num_candidates: torch.Tensor,
        mode: str = "train",
    ) -> None:
        """
        Update metrics with new batch of predictions and labels.

        Args:
            predictions: Model prediction tensor.
            weights: Sample weight tensor.
            labels: Ground truth label tensor.
            num_candidates: Number of candidates per sample (for GAUC).
            mode: Either 'train' or 'eval'.
        """
        # On eval, update BOTH the fresh set and the never-reset cumulative set
        # (if enabled) from the same batch.
        update_targets = list(self.all_metrics[mode])
        if mode == "eval" and "eval_cum" in self.all_metrics:
            update_targets = update_targets + self.all_metrics["eval_cum"]
        for metric in update_targets:
            if isinstance(metric, GAUCMetricComputation):
                metric.update(
                    predictions=predictions,
                    labels=labels,
                    weights=weights,
                    num_candidates=num_candidates,
                )
            else:
                metric.update(
                    predictions=predictions,
                    labels=labels,
                    weights=weights,
                )
        self.global_step[mode] += 1
        if mode == "train":
            # Accumulate on-device to avoid a per-step GPU->CPU sync; we read
            # the counter only at compute_and_log boundaries.
            self._perf_samples_counter += num_candidates.sum().to(
                self._perf_samples_counter.dtype
            )
            self._perf_steps_in_window += 1

    def compute(self, mode: str = "train") -> Dict[str, float]:
        """
        Compute and return all metrics for the current window.

        Args:
            mode: Either 'train' or 'eval'.

        Returns:
            Dictionary mapping metric names to their computed values.
        """
        all_computed_metrics = {}

        if mode == "eval" and "eval_cum" in self.all_metrics:
            # Dual-set eval: `window_*` (fresh per-pass) from the reset-each-pass
            # set; `lifetime_*` (cumulative across passes) from the never-reset
            # set. Filter each set to the matching prefix, and drop GAUC's
            # auxiliary `*_num_samples` reports. Key names are unchanged
            # (`window_auc`, `lifetime_ne`, ...) so dashboards keep working.
            def _emit(
                metrics: List[RecMetricComputation], keep_prefix: str
            ) -> None:
                for metric in metrics:
                    for computed in metric.compute():
                        pfx = str(computed.metric_prefix)
                        name = str(computed.name)
                        if pfx != keep_prefix or name.endswith("num_samples"):
                            continue
                        all_values = computed.value.cpu()
                        for i, task_name in enumerate(self.task_names):
                            if i >= len(all_values):
                                break
                            all_computed_metrics[f"metric/{pfx}{name}/{task_name}"] = (
                                all_values[i]
                            )

            _emit(self.all_metrics["eval"], "window_")
            _emit(self.all_metrics["eval_cum"], "lifetime_")
        else:
            for metric in self.all_metrics[mode]:
                computed_metrics = metric.compute()
                for computed in computed_metrics:
                    all_values = computed.value.cpu()
                    for i, task_name in enumerate(self.task_names):
                        if i >= len(all_values):
                            break
                        key = f"metric/{str(computed.metric_prefix) + str(computed.name)}/{task_name}"
                        all_computed_metrics[key] = all_values[i]

        logger.info(
            f"{mode} - Step {self.global_step[mode]} metrics: {all_computed_metrics}"
        )
        return all_computed_metrics

    def compute_and_log(
        self,
        mode: str = "train",
        additional_logs: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics and log to TensorBoard.

        Args:
            mode: Either 'train' or 'eval'.
            additional_logs: Optional additional data to log.

        Returns:
            Dictionary mapping metric names to their computed values.

        Raises:
            AssertionError: If TensorBoard logger is not configured.
        """
        assert self.tb_logger is not None
        all_computed_metrics = self.compute(mode)
        for k, v in all_computed_metrics.items():
            self.tb_logger.add_scalar(  # pyre-ignore [16]
                f"{mode}_{k}",
                v,
                global_step=self.global_step[mode],
            )

        if additional_logs is not None:
            for tag, data in additional_logs.items():
                for data_name, data_value in data.items():
                    self.tb_logger.add_scalar(
                        f"{tag}/{mode}_{data_name}",
                        data_value.detach().clone().cpu(),
                        global_step=self.global_step[mode],
                    )

        # Throughput metrics (train only). One GPU->CPU sync per call.
        if mode == "train" and self._perf_steps_in_window > 0:
            now = time.perf_counter()
            dt = max(now - self._perf_t_window, 1e-6)
            n_samples = int(self._perf_samples_counter.item())
            self._perf_total_samples += n_samples
            local_sps = n_samples / dt
            global_sps = local_sps * self._world_size
            step_ms = dt * 1000.0 / self._perf_steps_in_window
            elapsed = now - self._perf_t_start
            step = self.global_step["train"]
            self.tb_logger.add_scalar(
                "perf/train_samples_per_sec_local", local_sps, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_samples_per_sec_global", global_sps, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_step_time_ms", step_ms, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_total_samples", self._perf_total_samples, global_step=step
            )
            self.tb_logger.add_scalar(
                "perf/train_elapsed_sec", elapsed, global_step=step
            )
            # TFLOPS / MFU reporting (algo = dense yardstick, real = jagged).
            #   tflops_algo/gpu, mfu  — uses max_seq_len^2 attention work (the
            #     MFU yardstick: the FLOPs the workload would do if every
            #     user's UIH filled the padded seq length).
            #   tflops_real/gpu, hfu — uses this batch's mean(s_i^2) (actual
            #     GPU work; hardware utilization).
            #   fill                  — real / algo as a percent; how much of
            #     the algo budget the model actually executed this batch.
            # The jagged stash is read from the inner model; the model ref may
            # be a DMP wrapper, so unwrap via .module if present.
            tflops_str = ""
            if self._num_flops_per_sample > 0 and self._gpu_peak_flops > 0:
                local_flops = self._num_flops_per_sample * local_sps
                tflops_algo = local_flops / 1e12
                mfu = 100.0 * local_flops / self._gpu_peak_flops
                self.tb_logger.add_scalar("perf/train_tflops_algo_gpu", tflops_algo, global_step=step)
                self.tb_logger.add_scalar("perf/train_mfu_pct", mfu, global_step=step)
                tflops_str = f" tflops_algo/gpu={tflops_algo:.1f} mfu={mfu:.1f}%"
                jagged_t = None
                m = self._model_ref
                if m is not None:
                    inner = m.module if hasattr(m, "module") else m
                    jagged_t = getattr(inner, "_last_jagged_flops_per_sample", None)
                if jagged_t is not None:
                    jagged = float(jagged_t.item())
                    if 0 < jagged < self._num_flops_per_sample:
                        tflops_real = jagged * local_sps / 1e12
                        hfu = 100.0 * jagged * local_sps / self._gpu_peak_flops
                        fill = 100.0 * jagged / self._num_flops_per_sample
                        self.tb_logger.add_scalar("perf/train_tflops_real_gpu", tflops_real, global_step=step)
                        self.tb_logger.add_scalar("perf/train_hfu_pct", hfu, global_step=step)
                        self.tb_logger.add_scalar("perf/train_fill_pct", fill, global_step=step)
                        tflops_str += f" tflops_real/gpu={tflops_real:.1f} hfu={hfu:.1f}% fill={fill:.1f}%"
            logger.info(
                f"train - Step {step} perf: local_sps={local_sps:.1f} "
                f"global_sps={global_sps:.1f} step_ms={step_ms:.2f} "
                f"elapsed_sec={elapsed:.1f} total_samples={self._perf_total_samples}"
                + tflops_str
            )
            self._perf_t_window = now
            self._perf_steps_in_window = 0
            self._perf_samples_counter.zero_()

        # Time-to-target: latch wall-clock once any task's AUC crosses threshold.
        # Matches MLPerf DLRM-DCNv2 reporting style (default upstream target 0.80275).
        if (
            self._auc_threshold is not None
            and not self._time_to_target_logged
        ):
            for key, val in all_computed_metrics.items():
                metric_short = key.split("/")[-2] if "/" in key else key
                if metric_short.endswith("auc") and not metric_short.endswith("gauc"):
                    if float(val) >= self._auc_threshold:
                        ttt = time.perf_counter() - self._perf_t_start
                        self.tb_logger.add_scalar(
                            f"perf/time_to_auc_{self._auc_threshold:.5f}_sec",
                            ttt,
                            global_step=self.global_step[mode],
                        )
                        logger.info(
                            f"REACHED AUC>={self._auc_threshold} on {key}="
                            f"{float(val):.6f} at elapsed_sec={ttt:.2f} "
                            f"step={self.global_step[mode]}"
                        )
                        self._time_to_target_logged = True
                        break

        return all_computed_metrics

    def reset(self, mode: str = "train"):
        """
        Reset all metrics for a given mode.

        Args:
            mode: Either 'train' or 'eval'.
        """
        for metric in self.all_metrics[mode]:
            metric.reset()


# the datasets we support
SUPPORTED_DATASETS = [
    "debug",
    "movielens-1m",
    "movielens-20m",
    "movielens-13b",
    "movielens-18b",
    "kuairand-1k",
    "streaming-400m",
    "streaming-200b",
    "streaming-100b",
    "sampled-streaming-100b",
    "yambda-5b",
]


@gin.configurable
def env_path(key: str = "", default: str = "") -> str:
    """Resolve a path from os.environ[key], falling back to `default`.

    Intended as a gin macro so paths can be overridden via env vars without
    editing the gin file. Example gin usage:

        DATA_PATH = @env_path()
        env_path.key = "DLRM_DATA_PATH"
        env_path.default = "/some/default/path"
        make_train_test_dataloaders.new_path_prefix = %DATA_PATH
    """
    return os.environ.get(key, default) if key else default


@gin.configurable
def env_str(key: str = "", default: str = "") -> str:
    """Resolve a string from os.environ[key], falling back to `default`.

    Companion to `env_int`/`env_float` for categorical/string overrides (e.g. a
    metric backend selector). Example gin usage:

        MetricsLogger.train_lifetime_auc_mode = @tlam/env_str()
        tlam/env_str.key     = "TRAIN_LIFETIME_AUC_MODE"
        tlam/env_str.default = "binned"
    """
    raw = os.environ.get(key) if key else None
    return raw if raw else default


@gin.configurable
def env_int(key: str = "", default: int = 0) -> int:
    """Resolve an int from os.environ[key], falling back to `default`.

    Companion to `env_path` for numeric overrides. Example gin usage:

        make_optimizer_and_shard.hbm_cap_gb = @env_int()
        env_int.key = "HBM_CAP_GB"
        env_int.default = 260
    """
    raw = os.environ.get(key) if key else None
    return int(raw) if raw else default


@gin.configurable
def env_float(key: str = "", default: float = 0.0) -> float:
    """Resolve a float from os.environ[key], falling back to `default`.

    Companion to `env_int` for fractional/duration overrides (e.g. a
    checkpoint time interval in seconds). Example gin usage:

        streaming_train_eval_loop.checkpoint_time_interval_s = @env_float()
        env_float.key     = "CKPT_TIME_INTERVAL_S"
        env_float.default = 3600.0
    """
    raw = os.environ.get(key) if key else None
    return float(raw) if raw else default


_GPU_PEAK_FLOPS_TABLE: Dict[str, Dict[str, float]] = {
    # Per-GPU peak TFLOPS by dtype. Values from vendor datasheets / Primus-DLRM
    # peak_table. Used as the denominator in MFU/HFU. Keyed by case-insensitive
    # substring of torch.cuda.get_device_name(0).
    "MI355X": {"bf16": 2300e12, "fp32": 575e12},
    "MI350X": {"bf16": 2300e12, "fp32": 575e12},
    "MI300X": {"bf16": 1300e12, "fp32": 653e12},
    "MI325X": {"bf16": 1300e12, "fp32": 653e12},
    "B200":   {"bf16": 2250e12, "fp32": 1125e12},
    "H100":   {"bf16": 990e12,  "fp32": 67e12},
    "A100":   {"bf16": 312e12,  "fp32": 19.5e12},
}


def get_gpu_peak_flops(dtype: str = "bf16") -> float:
    """Peak FLOPS for the current GPU at the given dtype.

    Falls back to MI350X's number with a warning when the device name doesn't
    match any table entry — better to over-report MFU than to silently skip.
    """
    if not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name(0)
    for gpu_key, peaks in _GPU_PEAK_FLOPS_TABLE.items():
        if gpu_key in name:
            return peaks.get(dtype, peaks["bf16"])
    logger.warning(
        f"Unknown GPU for peak FLOPS: {name}; defaulting to MI350X bf16 (2300 TF)"
    )
    return _GPU_PEAK_FLOPS_TABLE["MI350X"]["bf16"]


@gin.configurable
def run_results_dir(run_name: str = "default", subdir: str = "results") -> str:
    """Resolve ``<recommendation_v4>/<subdir>/<run_name>`` from this file's location.

    Used as a gin macro to give per-run output directories that persist on the
    host (recommendation_v4 is bind-mounted into the training container).

    Example gin usage::

        RUN_NAME = @env_path()
        env_path.key     = "RUN_NAME"
        env_path.default = "default"
        run_results_dir.run_name = %RUN_NAME
        Profiler.trace_dir = @run_results_dir()
    """
    # utils.py lives at <recommendation_v4>/generative_recommenders/dlrm_v3/utils.py;
    # parents[2] climbs to <recommendation_v4>/.
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / subdir / run_name)


@gin.configurable
def get_dataset(
    name: str,
    new_path_prefix: str = "",
    history_length: Optional[int] = None,
    streaming_window_seconds: int = 86400,
    streaming_sort_within_window: bool = False,
    train_split_percentage: float = 1.0,
    split_salt: int = 0,
):
    """
    Get dataset class and configuration by name.

    Args:
        name: Dataset identifier (must be in SUPPORTED_DATASETS).
        new_path_prefix: Optional prefix to prepend to data paths.

    Returns:
        Tuple of (dataset_class, kwargs_dict) for dataset instantiation.

    Raises:
        AssertionError: If dataset name is not supported.
    """
    assert name in SUPPORTED_DATASETS, f"dataset {name} not supported"
    if name == "debug":
        return DLRMv3RandomDataset, {}
    if name == "movielens-1m":
        return (
            DLRMv3MovieLensDataset,
            {
                "ratings_file": os.path.join(
                    new_path_prefix, "data/ml-1m/sasrec_format.csv"
                ),
            },
        )
    if name == "movielens-20m":
        return (
            DLRMv3MovieLensDataset,
            {
                "ratings_file": os.path.join(
                    new_path_prefix, "data/ml-20m/sasrec_format.csv"
                ),
            },
        )
    if name == "movielens-13b":
        return (
            DLRMv3SyntheticMovieLensDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/ml-13b/16x16384"
                ),
            },
        )
    if name == "movielens-18b":
        return (
            DLRMv3SyntheticMovieLensDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/ml-18b/20x36864"
                ),
            },
        )
    if name == "kuairand-1k":
        return (
            DLRMv3KuaiRandDataset,
            {
                "seq_logs_file": os.path.join(
                    new_path_prefix, "data/KuaiRand-1K/data/processed_seqs.csv"
                ),
            },
        )
    if name == "streaming-400m":
        return (
            DLRMv3SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-400m/"
                ),
                "train_ts": 8,
                "total_ts": 10,
                "num_files": 3,
                "num_users": 150_000,
                "num_items": 1_500_000,
                "num_categories": 128,
            },
        )
    if name == "streaming-200b":
        return (
            DLRMv3SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-200b/"
                ),
                "train_ts": 90,
                "total_ts": 100,
                "num_files": 100,
                "num_users": 10_000_000,
                "num_items": 1_000_000_000,
                "num_categories": 128,
            },
        )
    if name == "streaming-100b":
        return (
            DLRMv3SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-100b/"
                ),
                "train_ts": 90,
                "total_ts": 100,
                "num_files": 100,
                "num_users": 5_000_000,
                "num_items": 1_000_000_000,
                "num_categories": 128,
            },
        )
    if name == "yambda-5b":
        from generative_recommenders.dlrm_v3.configs import YAMBDA_5B_CROSS_SPECS

        return (
            DLRMv3YambdaDataset,
            {
                # Layout: <new_path_prefix>/processed_5b/{train_sessions.parquet,...}
                # and <new_path_prefix>/shared_metadata/{artist,album}_item_mapping.parquet.
                # The dataset auto-builds a MAP_SHARED-mmap'd cache of the
                # flat columns + LISTEN-anchor positions under
                # <processed_dir>/hstu_cache_L<history_length>/ on first use;
                # all ranks on a node share the same physical pages.
                "processed_dir": os.path.join(new_path_prefix, "processed_5b"),
                "metadata_dir": os.path.join(new_path_prefix, "shared_metadata"),
                # Per-pool truncation cap; total interleaved UIH ~ 3*L/3 = L.
                # Override via `get_dataset.history_length = N` in gin.
                "history_length": history_length if history_length is not None else 4096,
                "scan_window": 20000,
                "cross_specs": YAMBDA_5B_CROSS_SPECS,
                # Temporal-streaming knobs (only used under --mode
                # streaming-train-eval; ignored by the default train-eval path).
                "streaming_window_seconds": streaming_window_seconds,
                "streaming_sort_within_window": streaming_sort_within_window,
                # User-level train:eval holdout for the streaming path. 1.0 =
                # no holdout (legacy). <1.0 holds out (1 - tsp) of users as a
                # fixed eval set; those users are never trained.
                "train_split_percentage": train_split_percentage,
                "split_salt": split_salt,
            },
        )
    if name == "sampled-streaming-100b":
        return (
            DLRMv3SyntheticStreamingDataset,
            {
                "ratings_file_prefix": os.path.join(
                    new_path_prefix, "data/streaming-100b/sampled_data/"
                ),
                "train_ts": 90,
                "total_ts": 100,
                "num_files": 1,
                "num_users": 50_000,
                "num_items": 1_000_000_000,
                "num_categories": 128,
            },
        )
