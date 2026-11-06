#!/usr/bin/env python3
"""Reproduce one selected CUDA forecast and all headline metrics after relocation."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent
ROOT = PACKAGE / "outputs/causal_v2"
sys.path.insert(0, str(PACKAGE))
from independent_tpp.protocol import sha256, write_json
from metrics import distribution_metrics

METRICS = ("min_ade_mean", "min_fde_mean", "mean_ade_mean", "mean_fde_mean",
           "energy_score_mean", "sample_diversity_mean")
KEYS = ("fold", "seed", "condition")


def recompute(row):
    with np.load(ROOT / row["prediction_path"]) as bundle:
        samples = bundle["samples"].astype(np.float64)
        truth = bundle["ground_truth"].astype(np.float64)
    distances = np.linalg.norm(samples - truth[:, None], axis=-1)
    ade, fde = distances.mean(axis=-1), distances[:, :, -1]
    scores = np.asarray([distribution_metrics(paths, label) for paths, label in zip(samples, truth)])
    numbers = [ade.min(axis=1).mean(), fde.min(axis=1).mean(),
               ade.mean(axis=1).mean(), fde.mean(axis=1).mean(), scores[:, 0].mean(), scores[:, 1].mean()]
    return {**{key: row[key] for key in KEYS}, **dict(zip(METRICS, numbers))}


def main():
    assert not (PROJECT / "third_party").exists(), "Run the release check in an isolated extraction"
    assert not (PROJECT / "trajectron").exists()
    manifest = PROJECT / "SHA256SUMS"
    files = 0
    for line in manifest.read_text().splitlines():
        digest, path = line.split("  ", 1)
        assert sha256(PROJECT / path) == digest, path
        files += 1
    subprocess.run([sys.executable, str(PACKAGE / "scripts/tiny_metric_check.py")], check=True)
    subprocess.run([sys.executable, str(PACKAGE / "own_unit_tests/test_causal_history.py")], check=True)
    index = pd.read_csv(ROOT / "prediction_index.csv")
    row = index.query("fold == 'eth' and seed == 0 and condition == 'clean_0'").iloc[0]
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "prediction.npz"
        subprocess.run([
            sys.executable, str(PACKAGE / "scripts/predict_independent.py"),
            "--fold", "eth", "--seed", "0", "--checkpoint", "selected",
            "--run-dir", str(ROOT / "runs/eth/seed_0"), "--device", "cuda",
            "--output", str(output),
        ], check=True, cwd=PROJECT)
        with np.load(output) as actual, np.load(ROOT / row.prediction_path) as expected:
            assert set(actual.files) == set(expected.files)
            for key in actual.files:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = pd.DataFrame(pool.map(recompute, index.to_dict(orient="records")))
    expected = pd.read_csv(ROOT / "results/group7_trajectronpp.csv")
    assert len(results) == len(expected) == 90
    reconstructed = results.merge(expected[[*KEYS, *METRICS]], on=list(KEYS), validate="one_to_one",
                                  suffixes=("", "_expected"))
    for metric in METRICS:
        np.testing.assert_allclose(reconstructed[metric], reconstructed[metric + "_expected"], atol=1e-10, rtol=1e-10)
    macro = results.groupby(["condition", "fold"])[list(METRICS)].mean().groupby("condition").mean()
    published = pd.read_csv(ROOT / "results/formal_statistical_macro.csv").pivot(
        index="condition", columns="metric", values="mean",
    )
    np.testing.assert_allclose(macro[list(METRICS)], published.loc[macro.index, list(METRICS)],
                               atol=1e-10, rtol=1e-10)
    results.to_csv(PROJECT / "reconstructed_results.csv", index=False)
    write_json(PROJECT / "reproduction_check.json", {
        "status": "passed", "hashes_verified": files, "exact_prediction": "ETH seed 0 clean",
        "recomputed_bundle_count": len(results), "metrics_per_bundle": list(METRICS),
        "macro_conditions": len(macro), "official_repository_present": False,
    })
    print(f"CAUSAL_RELEASE_REPRODUCED hashes={files} bundles={len(results)} exact_prediction=eth/seed_0/clean_0")


if __name__ == "__main__":
    main()
