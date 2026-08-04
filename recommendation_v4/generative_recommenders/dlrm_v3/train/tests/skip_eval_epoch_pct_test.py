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
"""Unit tests for `resolve_skip_eval_until_step` — the pure decision behind
SKIP_EVAL_EPOCH_PCT, which suppresses early periodic eval passes until the run
has trained a given fraction of the epoch.

These run without a GPU / distributed init: the streaming loop resolves the
threshold once at startup via this function and then does a plain
`global_step < threshold` comparison at each eval boundary, so exercising the
function directly covers the skip semantics.

The `test_production_gin_defaults_*` cases are the regression guard for the
per-GBS values documented in yambda_5b.gin: they replay the real eval grid and
assert eval starts on the intended boundary, using constants measured from the
20-seed RCP sweep (target AUC 0.75).
"""
import unittest

from generative_recommenders.dlrm_v3.train.utils import resolve_skip_eval_until_step

# --- constants from the yambda-5b RCP sweep ---------------------------------
# Epoch denominator (MLPerf TRAIN_SAMPLES) and the eval cadence in samples
# (EVAL_EVERY_DATA_PCT=0.001 of the epoch), both identical across GBS.
TOTAL_TRAIN_SAMPLES = 2290835423
EVAL_BLOCK_SAMPLES = 2293760

# Eval blocks of safety margin subtracted from the bare fit, so the start point
# sits strictly before the earliest crossing observed so far (that minimum can
# only move down as more seeds are collected).
MARGIN_EVALS = 2

# Per GBS: (gin SKIP_EVAL_EPOCH_PCT default, expected 1-based eval boundary the
# first eval lands on, earliest 0.75-crossing observed over the 20 seeds).
# The expected boundary must never be LATER than the earliest crossing, or a
# fast seed would converge unnoticed and be forced to train to the next
# boundary.
PRODUCTION_VALUES = {
    8192: (0.025, 25, 27),
    16384: (0.0303, 31, 34),
    32768: (0.041, 41, 43),
}


def _first_eval_boundary(pct: float, gbs: int) -> int:
    """Replay the real eval grid: return the 1-based ordinal of the first
    periodic eval boundary that is NOT skipped."""
    until_step = resolve_skip_eval_until_step(
        skip_eval_epoch_pct=pct,
        total_train_samples=TOTAL_TRAIN_SAMPLES,
        global_batch_size=gbs,
    )
    interval_steps = EVAL_BLOCK_SAMPLES // gbs
    ordinal = 1
    while ordinal * interval_steps < until_step:
        ordinal += 1
    return ordinal


class ResolveSkipEvalUntilStepTest(unittest.TestCase):
    def test_disabled_returns_zero(self) -> None:
        # 0.0 (and any non-positive) means OFF -> never skip.
        for pct in (0.0, -0.1):
            self.assertEqual(
                resolve_skip_eval_until_step(
                    skip_eval_epoch_pct=pct,
                    total_train_samples=TOTAL_TRAIN_SAMPLES,
                    global_batch_size=8192,
                ),
                0,
            )

    def test_degenerate_denominators_return_zero(self) -> None:
        # A missing epoch denominator (dataset without window_indices) or an
        # unset batch size must disable skipping rather than divide by zero.
        self.assertEqual(
            resolve_skip_eval_until_step(
                skip_eval_epoch_pct=0.025,
                total_train_samples=0,
                global_batch_size=8192,
            ),
            0,
        )
        self.assertEqual(
            resolve_skip_eval_until_step(
                skip_eval_epoch_pct=0.025,
                total_train_samples=TOTAL_TRAIN_SAMPLES,
                global_batch_size=0,
            ),
            0,
        )

    def test_rounds_up_so_eval_never_starts_early(self) -> None:
        # 1000 samples at gbs 300 -> 3.33 steps; must round UP to 4 so the
        # threshold is at or past the requested fraction, never before it.
        self.assertEqual(
            resolve_skip_eval_until_step(
                skip_eval_epoch_pct=0.1,
                total_train_samples=10000,
                global_batch_size=300,
            ),
            4,
        )
        # Exact division stays exact (no spurious +1).
        self.assertEqual(
            resolve_skip_eval_until_step(
                skip_eval_epoch_pct=0.5,
                total_train_samples=10000,
                global_batch_size=1000,
            ),
            5,
        )

    def test_threshold_marks_the_same_data_point_across_gbs(self) -> None:
        # The knob is a fraction of DATA, so doubling the global batch size must
        # halve the step threshold — the same sample count either way.
        pct = 0.02
        for gbs in (8192, 16384, 32768):
            until_step = resolve_skip_eval_until_step(
                skip_eval_epoch_pct=pct,
                total_train_samples=TOTAL_TRAIN_SAMPLES,
                global_batch_size=gbs,
            )
            samples = until_step * gbs
            target = pct * TOTAL_TRAIN_SAMPLES
            self.assertGreaterEqual(samples, target)
            # Overshoot is bounded by one step's worth of samples.
            self.assertLess(samples - target, gbs)

    def test_monotonic_in_pct(self) -> None:
        prev = -1
        for pct in (0.001, 0.01, 0.025, 0.05, 0.1, 0.5):
            cur = resolve_skip_eval_until_step(
                skip_eval_epoch_pct=pct,
                total_train_samples=TOTAL_TRAIN_SAMPLES,
                global_batch_size=8192,
            )
            self.assertGreater(cur, prev)
            prev = cur

    def test_production_gin_defaults_hit_expected_eval_boundary(self) -> None:
        for gbs, (pct, expected_ordinal, _) in PRODUCTION_VALUES.items():
            with self.subTest(gbs=gbs):
                self.assertEqual(_first_eval_boundary(pct, gbs), expected_ordinal)

    def test_production_gin_defaults_do_not_overshoot_convergence(self) -> None:
        # Starting eval LATER than the earliest observed 0.75 crossing would
        # force a fast seed to keep training past its stopping point.
        for gbs, (pct, _, earliest_crossing) in PRODUCTION_VALUES.items():
            with self.subTest(gbs=gbs):
                self.assertLessEqual(
                    _first_eval_boundary(pct, gbs), earliest_crossing
                )

    def test_gin_defaults_track_the_documented_formula(self) -> None:
        # The gin defaults must stay in sync with the margined fit documented in
        # the README and the gin comment:
        #   start_eval_samples = FLOOR((4480*GBS + 135331840) / 3)
        # where the 2-eval-block margin is already folded into the intercept
        # (149094400 - 3*MARGIN_EVALS*EVAL_BLOCK == 135331840).
        # They are written to a few significant figures, so require the
        # configured value to sit at or just BELOW the exact value (rounding
        # down starts eval a touch early, which is safe; rounding up could eat
        # into the margin) and within one eval block of it.
        block_pct = EVAL_BLOCK_SAMPLES / TOTAL_TRAIN_SAMPLES
        for gbs, (pct, _, _) in PRODUCTION_VALUES.items():
            with self.subTest(gbs=gbs):
                start_eval_samples = (4480 * gbs + 135331840) // 3
                exact = start_eval_samples / TOTAL_TRAIN_SAMPLES
                self.assertLessEqual(pct, exact)
                self.assertGreater(pct, exact - block_pct)

    def test_margin_moves_start_exactly_two_evals_earlier(self) -> None:
        # The margin is defined in whole eval blocks, so each configured value
        # must land exactly MARGIN_EVALS boundaries before the bare fit would.
        # 149094400 here is the PRE-margin intercept, i.e. the raw two-point fit
        # through the observed crossings, not the 135331840 used in production.
        for gbs, (pct, _, _) in PRODUCTION_VALUES.items():
            with self.subTest(gbs=gbs):
                bare_pct = ((4480 * gbs + 149094400) // 3) / TOTAL_TRAIN_SAMPLES
                self.assertEqual(
                    _first_eval_boundary(pct, gbs),
                    _first_eval_boundary(bare_pct, gbs) - MARGIN_EVALS,
                )


if __name__ == "__main__":
    unittest.main()
