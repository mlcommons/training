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

# pyre-strict

"""
Round-trip correctness test for ``LifetimeAUCMetricComputation`` checkpoint
serialization.

Background: torchrec's ``AUCMetricComputation`` registers its
PREDICTIONS/LABELS/WEIGHTS buffers with ``persistent=False``, so the default
``state_dict()`` returns them empty and a separate ``_num_samples`` counter is
dropped too. Without the overrides on ``LifetimeAUCMetricComputation`` every
checkpoint resume would silently restart the lifetime AUC from an empty buffer.

These tests assert:
  1. update -> compute == A; state_dict -> load_state_dict on a fresh metric ->
     compute == A (buffers survive the round trip).
  2. ``_num_samples`` round-trips exactly (required so the next update() does
     not take the init-sentinel branch and desync windowed eviction).
  3. The shared-blob path (buffers stripped) leaves a fresh metric empty, so the
     per-rank artifact is the sole authority for the trailing buffer.

Runs in <1s on CPU. Skipped automatically if torchrec is unavailable.
"""

import unittest

import torch

try:
    from generative_recommenders.dlrm_v3.utils import LifetimeAUCMetricComputation

    _HAVE_DEPS = True
except Exception:  # pragma: no cover - import guard for envs without torchrec
    _HAVE_DEPS = False


def _make_metric(n_tasks: int = 1, window: int = 10_000_000):
    return LifetimeAUCMetricComputation(
        my_rank=0,
        batch_size=128,
        n_tasks=n_tasks,
        window_size=window,
    )


def _feed(metric, preds, labels, weights) -> None:
    metric.update(
        predictions=preds,
        labels=labels,
        weights=weights,
    )


@unittest.skipUnless(_HAVE_DEPS, "torchrec / generative_recommenders not importable")
class LifetimeAUCResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.n_tasks = 1
        self.n = 4096
        self.preds = torch.rand(self.n_tasks, self.n)
        self.labels = (torch.rand(self.n_tasks, self.n) > 0.5).float()
        self.weights = torch.ones(self.n_tasks, self.n)

    def _compute_value(self, metric) -> float:
        reports = metric._compute()
        return float(reports[0].value.flatten()[0].item())

    def test_state_dict_round_trip_preserves_auc(self) -> None:
        m = _make_metric(self.n_tasks)
        _feed(m, self.preds, self.labels, self.weights)
        auc_a = self._compute_value(m)
        n_a = m.lifetime_sample_count()
        self.assertEqual(n_a, self.n)

        sd = m.state_dict()

        fresh = _make_metric(self.n_tasks)
        fresh.load_state_dict(sd)
        auc_b = self._compute_value(fresh)

        self.assertEqual(fresh.lifetime_sample_count(), self.n)
        self.assertAlmostEqual(auc_a, auc_b, places=6)

    def test_num_samples_round_trips(self) -> None:
        m = _make_metric(self.n_tasks)
        _feed(m, self.preds, self.labels, self.weights)
        sd = m.state_dict()
        fresh = _make_metric(self.n_tasks)
        fresh.load_state_dict(sd)
        self.assertEqual(fresh._num_samples, m._num_samples)

    def test_continued_update_after_resume_matches_uninterrupted(self) -> None:
        # Splitting a stream and resuming in the middle must equal feeding it all
        # at once (this is what fails when _num_samples is not restored).
        half = self.n // 2
        p1, p2 = self.preds[:, :half], self.preds[:, half:]
        l1, l2 = self.labels[:, :half], self.labels[:, half:]
        w1, w2 = self.weights[:, :half], self.weights[:, half:]

        ref = _make_metric(self.n_tasks)
        _feed(ref, p1, l1, w1)
        _feed(ref, p2, l2, w2)
        auc_ref = self._compute_value(ref)

        part = _make_metric(self.n_tasks)
        _feed(part, p1, l1, w1)
        resumed = _make_metric(self.n_tasks)
        resumed.load_state_dict(part.state_dict())
        _feed(resumed, p2, l2, w2)
        auc_resumed = self._compute_value(resumed)

        self.assertAlmostEqual(auc_ref, auc_resumed, places=6)

    def test_blob_state_dict_strips_buffers(self) -> None:
        from generative_recommenders.dlrm_v3.checkpoint import (
            _metric_blob_state_dict,
        )

        m = _make_metric(self.n_tasks)
        _feed(m, self.preds, self.labels, self.weights)
        blob = _metric_blob_state_dict(m)
        prefix = LifetimeAUCMetricComputation._LIFETIME_KEY_PREFIX
        self.assertFalse(any(k.startswith(prefix) for k in blob.keys()))

        # A fresh metric loaded from the stripped blob must NOT have history —
        # the per-rank artifact is the only source of the trailing buffer.
        fresh = _make_metric(self.n_tasks)
        fresh.load_state_dict(blob)
        self.assertEqual(fresh.lifetime_sample_count(), 0)


if __name__ == "__main__":
    unittest.main()
