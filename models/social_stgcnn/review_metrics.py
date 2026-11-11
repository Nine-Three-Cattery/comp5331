"""Recheck displacement errors and compute heading diagnostics on the frozen forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from li_junhao_trajectronpp.independent_tpp.data import load_split
from li_junhao_trajectronpp.independent_tpp.protocol import CONDITIONS, FOLDS, SEEDS
from li_junhao_trajectronpp.scripts.analyze_independent_diagnostics import (
    heading_rows, prediction_path, training_thresholds,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRIBUTION = ROOT / "submission/contributions/social_stgcnn"


def displacement_errors(samples, truth):
    """Average over windows last, choosing the best draw separately for ADE and FDE."""
    offsets = np.subtract(samples, truth[:, None], dtype=np.float64)
    distances = np.hypot(offsets[..., 0], offsets[..., 1])
    ade = distances.mean(axis=-1)
    fde = distances[..., -1]
    return {
        "min_ade": ade.min(axis=1).mean(),
        "min_fde": fde.min(axis=1).mean(),
        "mean_ade": ade.mean(axis=1).mean(),
        "mean_fde": fde.mean(axis=1).mean(),
    }


def review(runs, data, results):
    reference = pd.read_csv(results).set_index(["fold", "seed", "condition"], verify_integrity=True)
    rows = []
    for fold in FOLDS:
        threshold = training_thresholds(load_split(fold, "train", str(data)))["turn_threshold_rad"]
        for seed in SEEDS:
            for condition, _, _ in CONDITIONS:
                key = (fold, seed, condition)
                scored = reference.loc[key]
                path = prediction_path(runs / fold / f"seed_{seed}", condition)
                with np.load(path, allow_pickle=False) as bundle:
                    identity = json.loads(str(bundle["metadata"]))
                    if any(identity[k] != scored[k] for k in ("manifest_hash", "checkpoint_sha256")):
                        raise ValueError(f"The forecasts and score table identify different runs at {key}")
                    samples = bundle["samples"]
                    if samples.shape[:2] != (scored.window_count, scored.sample_count):
                        raise ValueError(f"Window or draw count differs at {key}")
                    errors = displacement_errors(samples, bundle["ground_truth"])
                    heading = heading_rows({"samples": samples, "observed": bundle["observed"]},
                                           None, threshold, fold, seed, condition)
                published = np.array([scored[f"{metric}_mean"] for metric in errors])
                measured = np.array(list(errors.values()))
                np.testing.assert_allclose(measured, published, rtol=0, atol=1e-7, err_msg=str(key))
                rows.append({
                    "fold": fold, "seed": seed, "condition": condition, "windows": len(samples),
                    **errors, "max_score_difference_m": np.abs(measured - published).max(),
                    "training_turn_p95_rad": threshold,
                    "mean_abs_heading_change_rad": heading["turn_rate_mean_mean"],
                    "windows_reaching_turn_threshold_fraction": heading["turn_violation_rate_mean"],
                })
        print(f"Completed {fold}", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts/social_stgcnn/formal/runs")
    parser.add_argument("--data", type=Path, default=ROOT / "data/causal_v2")
    parser.add_argument("--results", type=Path, default=CONTRIBUTION / "results.csv")
    parser.add_argument("--out", type=Path, default=CONTRIBUTION / "evidence/review_metrics.csv")
    args = parser.parse_args()
    table = review(args.runs, args.data, args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"Reviewed {len(table)} bundles. Largest displacement-score difference: "
          f"{table.max_score_difference_m.max():.3g} m")


if __name__ == "__main__":
    main()
