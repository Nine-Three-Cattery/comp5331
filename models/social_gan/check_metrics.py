"""Independently calculate the six causal_v2 scores from the exported forecasts."""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTRIBUTION = ROOT / "submission/contributions/social_gan"
sys.path[:0] = [str(ROOT / "benchmark"), str(ROOT / "li_junhao_trajectronpp")]
from contract import FIXTURES, fingerprint, sha256
from independent_tpp.protocol import CONDITIONS, SEEDS, digest_json, write_json


METRICS = ("min_ade", "min_fde", "mean_ade", "mean_fde", "energy_score", "sample_diversity")
REL_TOL, ABS_TOL = 1e-7, 1e-8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def calculate(samples, truth):
    """Return fold means with separate per-window ADE and FDE minima."""
    k = samples.shape[1]
    left, right = np.triu_indices(k, 1)
    totals = np.zeros(len(METRICS), dtype=np.float64)
    for start in range(0, len(truth), 256):
        batch = samples[start:start + 256].astype(np.float64)
        difference = batch - truth[start:start + 256, None]
        squared = np.square(difference).sum(axis=-1)
        distances = np.sqrt(squared)
        ade = distances.mean(axis=-1)
        fde = distances[:, :, -1]
        if k > 1:
            pairwise = batch[:, left] - batch[:, right]
            diversity = np.sqrt(np.square(pairwise).sum(axis=-1).mean(axis=-1)).mean(axis=1)
        else:
            diversity = np.zeros(len(batch), dtype=np.float64)
        energy = np.sqrt(squared.mean(axis=-1)).mean(axis=1) - diversity / 2
        totals += np.array([ade.min(axis=1).sum(), fde.min(axis=1).sum(),
                            ade.mean(axis=1).sum(), fde.mean(axis=1).sum(),
                            energy.sum(), diversity.sum()])
    return dict(zip(METRICS, (totals / len(truth)).tolist()))


def fixture():
    # Draw A has errors (0, 2) and draw B has errors (3, 0).
    # Their trajectory RMS distance is sqrt((3^2 + 2^2) / 2).
    truth = np.zeros((1, 2, 2), dtype=np.float32)
    samples = np.array([[[[0, 0], [2, 0]], [[0, 3], [0, 0]]]], dtype=np.float32)
    diversity = math.sqrt(6.5)
    known = {1: (1., 2., 1., 2., math.sqrt(2), 0.),
             2: (1., 0., 1.25, 1., (math.sqrt(2) + math.sqrt(4.5) - diversity) / 2, diversity)}
    for k, expected in known.items():
        np.testing.assert_allclose(list(calculate(samples[:, :k], truth).values()), expected,
                                   rtol=0, atol=1e-14, err_msg=f"fixture K={k}")
    return {f"K{k}": dict(zip(METRICS, values)) for k, values in known.items()}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(runs, data, results):
    lock = read_json(FIXTURES / "manifests.json")
    rows = read_csv(results)
    indexed = {(row["fold"], int(row["seed"]), row["condition"]): row for row in rows}
    conditions = {tag: (kind, level) for tag, kind, level in CONDITIONS}
    expected = set(itertools.product(lock["folds"], SEEDS, conditions))
    require(len(rows) == len(expected) and set(indexed) == expected, "Incomplete formal matrix")
    max_errors = dict.fromkeys(METRICS, 0.)
    counts = {"windows_scored": 0, "trajectories_scored": 0}
    for fold, splits in lock["folds"].items():
        manifest = data / fold / "test_manifest.csv"
        test_hash = sha256(manifest)
        window_ids = [row["window_id"] for row in read_csv(manifest)]
        n = len(window_ids)
        require(test_hash == splits["test"]["sha256"] and n == splits["test"]["window_count"],
                f"{fold}: manifest does not match the lock")
        reference_truth = None
        for seed in SEEDS:
            run = runs / fold / f"seed_{seed}"
            config_hash = digest_json(read_json(run / "config.json"))
            selection = read_json(run / "checkpoint_selection.json")
            checkpoint = selection["selected"]["checkpoint"]
            require(checkpoint == Path(checkpoint).name and selection["selection_split"] == "val"
                    and selection["test_used_for_selection"] is False, f"{fold}/{seed}: invalid checkpoint selection")
            checkpoint_hash = sha256(run / checkpoint)
            for tag, (kind, level) in conditions.items():
                label = f"{fold}/{seed}/{tag}"
                row = indexed[fold, seed, tag]
                path = run / "evaluation" / tag / f"predictions_test_{tag}.npz"
                identity = {"manifest_hash": test_hash, "config_hash": config_hash,
                            "checkpoint_sha256": checkpoint_hash}
                expected_row = {**identity, "protocol": lock["protocol"], "sample_count": "20",
                                "window_count": str(n), "prediction_sha256": sha256(path)}
                require(all(row[key] == value for key, value in expected_row.items()), f"{label}: CSV identity mismatch")
                with np.load(path, allow_pickle=False) as bundle:
                    metadata = json.loads(str(bundle["metadata"]))
                    expected_metadata = {**identity, "version": lock["protocol"], "fold": fold, "split": "test",
                                         "seed": seed, "sample_count": 20, "condition": kind, "level": level,
                                         "checkpoint": checkpoint}
                    require(all(metadata[key] == value for key, value in expected_metadata.items()),
                            f"{label}: prediction identity mismatch")
                    require([json.loads(record)["window_id"] for record in bundle["records"]] == window_ids,
                            f"{label}: manifest order mismatch")
                    samples, truth = bundle["samples"], bundle["ground_truth"]
                require(samples.shape == (n, 20, 12, 2) and truth.shape == (n, 12, 2)
                        and np.isfinite(samples).all() and np.isfinite(truth).all(),
                        f"{label}: invalid prediction or truth coordinates")
                if reference_truth is None:
                    reference_truth = truth
                require(np.array_equal(truth, reference_truth), f"{label}: ground truth changed within fold")
                for metric, value in calculate(samples, truth).items():
                    reported = float(row[f"{metric}_mean"])
                    require(math.isclose(value, reported, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                            f"{label}: {metric} {value} differs from reported {reported}")
                    max_errors[metric] = max(max_errors[metric], abs(value - reported))
                counts["windows_scored"] += n
                counts["trajectories_scored"] += n * 20
        print(f"OWNER_METRICS_FOLD_OK {fold}", flush=True)
    return {
        "model": "social_gan", "protocol": lock["protocol"], "status": "passed",
        "source_sha256": sha256(Path(__file__)), "contract_sha256": fingerprint(),
        "results_sha256": sha256(results), "prediction_bundles": len(rows),
        "metric_values_compared": len(rows) * len(METRICS), **counts,
        "max_absolute_difference": max_errors, "relative_tolerance": REL_TOL, "absolute_tolerance": ABS_TOL,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts/social_gan/formal/runs")
    parser.add_argument("--data", type=Path, default=ROOT / "data/causal_v2")
    parser.add_argument("--results", type=Path, default=CONTRIBUTION / "results.csv")
    parser.add_argument("--output", type=Path, default=CONTRIBUTION / "evidence/independent_metrics.json")
    parser.add_argument("--fixture-only", action="store_true")
    args = parser.parse_args()
    known = fixture()
    print("OWNER_METRICS_FIXTURE_OK K=1,2", flush=True)
    if not args.fixture_only:
        receipt = verify(args.runs, args.data, args.results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {**receipt, "fixture_expected": known})
        print(f"OWNER_METRICS_OK {receipt['prediction_bundles']} bundles and {receipt['metric_values_compared']} comparisons")


if __name__ == "__main__":
    main()
