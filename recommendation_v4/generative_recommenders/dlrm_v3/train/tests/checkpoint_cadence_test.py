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
"""Unit tests for `select_in_window_checkpoint_reason` — the pure decision that
drives the streaming loop's three fine-grained checkpoint cadences:

  * `in_window_checkpoint_frequency` — per-window-local batch count
  * `checkpoint_step_frequency`      — monotonic global step ("every 1000 steps")
  * `checkpoint_time_interval_s`     — wall-clock ("hourly")

These run without a GPU / distributed init: the loop broadcasts a single
`elapsed_since_last_save` from rank 0 and then calls this pure function, so
exercising the function directly fully covers the trigger semantics.
"""
import unittest

from generative_recommenders.dlrm_v3.train.utils import (
    select_in_window_checkpoint_reason,
)


def _reason(
    *,
    batch: int = 1,
    step: int = 1,
    elapsed: float = 0.0,
    in_window: int = 0,
    step_freq: int = 0,
    time_s: float = 0.0,
) -> str | None:
    return select_in_window_checkpoint_reason(
        train_batch_idx=batch,
        global_step=step,
        elapsed_since_last_save=elapsed,
        in_window_checkpoint_frequency=in_window,
        checkpoint_step_frequency=step_freq,
        checkpoint_time_interval_s=time_s,
    )


class CheckpointCadenceTest(unittest.TestCase):
    def test_all_disabled_never_fires(self) -> None:
        for batch in (1, 100, 1000):
            for step in (1, 1000, 5000):
                self.assertIsNone(_reason(batch=batch, step=step, elapsed=1e9))

    def test_step_based_every_1000(self) -> None:
        # Fires exactly on multiples of the step frequency.
        self.assertEqual(_reason(step=1000, step_freq=1000), "global_step")
        self.assertEqual(_reason(step=2000, step_freq=1000), "global_step")
        # Does not fire just off a boundary.
        self.assertIsNone(_reason(step=999, step_freq=1000))
        self.assertIsNone(_reason(step=1001, step_freq=1000))

    def test_step_zero_does_not_trigger(self) -> None:
        # global_step==0 must not trivially satisfy `0 % N == 0`.
        self.assertIsNone(_reason(step=0, step_freq=1000))

    def test_time_based_interval(self) -> None:
        # At/over the interval -> fires; under -> no save.
        self.assertEqual(
            _reason(step=3, elapsed=3600.0, time_s=3600.0), "time_interval"
        )
        self.assertEqual(
            _reason(step=3, elapsed=4000.0, time_s=3600.0), "time_interval"
        )
        self.assertIsNone(_reason(step=3, elapsed=3599.9, time_s=3600.0))

    def test_in_window_batch_cadence(self) -> None:
        self.assertEqual(_reason(batch=5, in_window=5), "in_window_batch")
        self.assertEqual(_reason(batch=10, in_window=5), "in_window_batch")
        self.assertIsNone(_reason(batch=4, in_window=5))

    def test_precedence_in_window_over_step_over_time(self) -> None:
        # All three would fire this batch; precedence picks in_window first.
        self.assertEqual(
            _reason(
                batch=5,
                step=1000,
                elapsed=9999.0,
                in_window=5,
                step_freq=1000,
                time_s=3600.0,
            ),
            "in_window_batch",
        )
        # in_window not due this batch -> step wins over time.
        self.assertEqual(
            _reason(
                batch=4,
                step=1000,
                elapsed=9999.0,
                in_window=5,
                step_freq=1000,
                time_s=3600.0,
            ),
            "global_step",
        )
        # Neither batch nor step due -> time wins.
        self.assertEqual(
            _reason(
                batch=4,
                step=999,
                elapsed=9999.0,
                in_window=5,
                step_freq=1000,
                time_s=3600.0,
            ),
            "time_interval",
        )

    def test_step_and_time_combined_independent(self) -> None:
        # Step frequency enabled, time disabled: only step boundaries fire.
        self.assertEqual(_reason(step=1000, step_freq=1000, time_s=0.0), "global_step")
        self.assertIsNone(_reason(step=1000, elapsed=1e9, step_freq=0, time_s=0.0))


if __name__ == "__main__":
    unittest.main()
