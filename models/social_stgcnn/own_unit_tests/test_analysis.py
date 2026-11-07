"""Synthetic checks for evidence joins, denominators, and sample-prefix selection."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from models.social_stgcnn.analyze import (
    MOTION, NEIGHBOR_BINS, SCORES, STAGES, clean_evidence, runtime_row,
    scored_windows, summarize_runs, summarize_runtime,
)


class EvidenceTests(unittest.TestCase):
    def test_pair_by_id_and_reject_missing_target(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload = {"window_id": ["second", "first"], **{k: [3., 1.] for k in [*SCORES, *MOTION]}}
            pd.DataFrame(payload).to_csv(directory / "window_metrics.csv", index=False)
            frame = scored_windows(directory, pd.Index(["first", "second"], name="window_id"))
            self.assertEqual(list(frame.mean_ade), [1., 3.])
            with self.assertRaisesRegex(ValueError, "test manifest"):
                scored_windows(directory, pd.Index(["first", "other"], name="window_id"))

    def test_equal_folds_unique_denominators_and_absent_neighbor_bins(self):
        rows = []
        for seed in (0, 1, 2):
            for fold, dense_count, base in (("eth", 1, 1.), ("univ", 20, 5.)):
                rows.append(dict(fold=fold, seed=seed, bin="8+", unique_windows=dense_count, ade=base + seed))
                rows.append(dict(fold=fold, seed=seed, bin="0", unique_windows=int(fold == "eth"),
                                 ade=2. + seed if fold == "eth" else np.nan))
        output = summarize_runs(rows, ["bin"], ["ade"], seed_spread=["ade"])
        macro = output.loc[output.scope == "macro"].set_index("bin")
        self.assertEqual(macro.loc["8+", "ade"], 4.)
        self.assertEqual(macro.loc["8+", "ade_seed_sd"], 1.)
        self.assertEqual(macro.loc["8+", "unique_windows"], 21)
        self.assertEqual(macro.loc["0", "ade"], 3.)
        self.assertEqual(macro.loc["0", "contributing_folds"], 1)
        empty = output.loc[(output.fold == "univ") & (output.bin == "0")].iloc[0]
        self.assertEqual(empty.unique_windows, 0)
        self.assertTrue(np.isnan(empty.ade))

    def test_prefixes_use_same_draws_and_worst_window_ties_use_id(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            location = run / "evaluation/clean_0"
            location.mkdir(parents=True)
            samples = np.zeros((2, 20, 12, 2), dtype=np.float32)
            samples[:, :, :, 0] = np.arange(20, 0, -1)[:, None]
            target = np.zeros((2, 12, 2), dtype=np.float32)
            reference = {key: "fixture" for key in ("config_hash", "checkpoint_sha256", "manifest_hash")}
            np.savez(location / "predictions_test_clean_0.npz", samples=samples, ground_truth=target,
                     observed=np.zeros((2, 8, 2)), metadata=json.dumps(reference))
            clean = pd.DataFrame({name: [1., 1.] for name in SCORES}, index=["window-b", "window-a"])
            budgets, cases = clean_evidence(run, "hotel", 0, clean, np.array([0, 2]), reference)
            self.assertEqual([row["min_ade"] for row in budgets], [20., 16., 11., 1.])
            case, paths = cases[0]
            self.assertEqual(case["window_id"], "window-a")
            self.assertEqual(case["available_and_used_neighbors"], 2)
            self.assertEqual(len(paths), 8 + 12 + 20 * 12)

    def test_runtime_keeps_hosts_separate_and_marks_untimed_bins(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            medians = dict(zip(STAGES, (2., 3., 6.)))
            primary = {"targets": 128, "batch_size": 32, "sample_count": 20, "warmups": 3,
                       "measured_repetitions": 10, "repetitions": [medians] * 10, "medians": medians}
            density = [{"available_neighbors_bin": name, "available_windows": 10, "timed_windows": 0}
                       for name in NEIGHBOR_BINS]
            density[-1].update(available_windows=64, timed_windows=32, repetitions=[medians] * 10,
                               medians=medians)
            hosts = {}
            records = []
            for private_name in ("first.private", "second.private"):
                timing = {**primary, "environment": {"host": private_name, "device": "cpu"}, "density": density}
                (run / "runtime.json").write_text(json.dumps(timing))
                record, cells = runtime_row(run, "eth", 0, hosts)
                self.assertNotIn(private_name, str((record, cells)))
                self.assertTrue(np.isnan(cells[0]["end_to_end_ms"]))
                self.assertEqual(cells[-1]["end_to_end_ms"], 6.)
                records.append(record)
            overall = summarize_runtime(records)
            matched = overall.loc[overall.scope != "run"]
            self.assertEqual(set(matched.host), {"Host 1", "Host 2"})
            self.assertEqual(matched.runs.tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
