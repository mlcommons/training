"""CPU-only coverage for the streaming evaluation metrics reader."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from recommendation.scripts.build_ne_auc_trajectory import parse_log, parse_metrics_jsonl


class MetricsJsonlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "run.metrics.jsonl"

    def write_records(self, records):
        self.path.write_text("\n".join(json.dumps(row) for row in records) + "\n")

    def test_task_selection_and_global_steps(self):
        self.write_records([
            {"global_step": 201, "metric/window_auc/listen_plus": 0.71,
             "metric/window_ne/listen_plus": 0.94, "metric/window_auc/like": 0.8},
            {"global_step": 402, "train_ts": None,
             "metric/window_auc/listen_plus": 0.75,
             "metric/window_ne/listen_plus": float("nan")},
        ])
        traj, perf = parse_metrics_jsonl(str(self.path), "listen_plus")
        self.assertEqual(traj["train"], {})
        self.assertEqual(perf, [])
        self.assertEqual(set(traj["eval"]), {201, 402})
        self.assertEqual(traj["eval"][201], {"window_auc": 0.71, "window_ne": 0.94})
        self.assertTrue(math.isnan(traj["eval"][402]["window_ne"]))

    def test_last_matching_record_wins_at_the_original_step(self):
        self.write_records([
            {"global_step": 201, "metric/window_auc/listen_plus": 0.71},
            {"global_step": 201, "train_ts": None,
             "metric/window_auc/listen_plus": 0.75},
            {"global_step": 201, "metric/window_auc/like": 0.8},
        ])
        traj, _ = parse_metrics_jsonl(str(self.path), "listen_plus")
        self.assertEqual(traj["eval"], {201: {"window_auc": 0.75}})

    def test_invalid_or_unmatched_input_is_rejected(self):
        for text in ("{broken", "{}", "", "null", json.dumps({
            "global_step": 1, "metric/window_auc/like": 0.8,
        })):
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaises(ValueError):
                    parse_metrics_jsonl(str(self.path), "listen_plus")

    def test_text_log_still_uses_tensor_metrics(self):
        self.path.write_text(
            "train - Step 201 metrics: {'metric/window_auc/listen_plus': tensor(0.71)}\n"
        )
        traj, _ = parse_log(str(self.path), "listen_plus")
        self.assertEqual(traj["train"], {201: {"window_auc": 0.71}})


if __name__ == "__main__":
    unittest.main()
