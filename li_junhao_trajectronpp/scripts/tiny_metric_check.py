from __future__ import print_function

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


sys.path.insert(0, os.path.dirname(__file__))
from metrics import compute_window_metrics  # noqa: E402


def check_bundle_evaluator():
    with tempfile.TemporaryDirectory(prefix="causal-evaluator-") as directory:
        root = Path(directory)
        target = np.zeros((2, 12, 2))
        observed = np.zeros((2, 8, 2))
        records = np.array([
            json.dumps({
                "fold": "eth", "split": "test", "scene_id": "eth",
                "scene_name": "fixture", "scene_t": 7, "anchor_t": 7,
                "source_file": "fixture", "target_id": str(index),
                "window_id": f"fixture:{index}:7", "condition": "clean",
                "sigma": 0.0, "missing": 0, "dt": 0.4,
            }) for index in (1, 2)
        ])
        for count in (1, 20):
            samples = np.repeat(target[:, None], count, axis=1)
            samples[:, count // 2:, :, 0] = 1
            bundle = root / f"k{count}.npz"
            output = root / f"k{count}_metrics"
            np.savez(bundle, samples=samples, ground_truth=target, observed=observed, records=records)
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("evaluate.py")),
                "--predictions", str(bundle), "--output-dir", str(output),
            ], stdout=subprocess.DEVNULL, check=True)
            result = json.loads((output / "summary.json").read_text())
            assert result["sample_count"] == count
            assert np.isclose(result["overall"]["min_ade_mean"], float(count == 1))
            assert np.isclose(result["overall"]["mean_ade_mean"], 1 if count == 1 else 0.5)
            print(f"COMMON_EVALUATOR_OK K={count}")


def main():
    target = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    samples = np.asarray(
        [
            target,
            target + np.asarray([1.0, 0.0]),
        ]
    )
    result = compute_window_metrics(samples, target, np.asarray([0.0, 0.0]), 0.4)
    expected = {
        "min_ade": 0.0,
        "mean_ade": 0.5,
        "min_fde": 0.0,
        "mean_fde": 0.5,
        "energy_score": 0.0,
        "sample_diversity": 1.0,
        "sample_count": 2,
    }
    for key, value in expected.items():
        if not np.isclose(result[key], value):
            raise AssertionError("{}: {} != {}".format(key, result[key], value))
    for count, offset_count in ((1, 1), (20, 10)):
        fixture = np.repeat(target[None], count, axis=0)
        fixture[-offset_count:, :, 0] += 1
        actual = compute_window_metrics(fixture, target, np.zeros(2), 0.4)
        expected = {
            "min_ade": float(count == 1),
            "min_fde": float(count == 1),
            "mean_ade": offset_count / count,
            "mean_fde": offset_count / count,
            "sample_diversity": 10 / 19 if count == 20 else 0,
            "energy_score": 9 / 38 if count == 20 else 1,
        }
        for key, value in expected.items():
            if not np.isclose(actual[key], value):
                raise AssertionError(f"K={count} {key}: {actual[key]} != {value}")
    print("TINY_METRIC_REFERENCE_OK K=1,2,20")
    check_bundle_evaluator()


if __name__ == "__main__":
    main()
