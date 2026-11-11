#!/usr/bin/env python3
"""Verify the causal matrix and export a relocatable prediction index."""

import argparse
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))
from independent_tpp.data import CachedNeutralDataset, cache_path, load_split
from independent_tpp.protocol import (
    CONDITIONS, FOLDS, SEEDS, VERSION, corruption_seed, digest_json, perturb_history, sha256,
    write_json,
)


from train_independent import source_fingerprint


EXPECTED_SOURCE = source_fingerprint()


def read(path):
    return json.loads(path.read_text())


def check_imports():
    roots = [PACKAGE / "independent_tpp"]
    paths = [path for root in roots for path in root.glob("*.py")]
    paths.extend(PACKAGE / "scripts" / name for name in (
        "prepare_neutral_data.py", "cache_independent_data.py", "train_independent.py",
        "select_independent_checkpoint.py", "predict_independent.py",
        "evaluate_independent_run.py", "evaluate.py", "metrics.py",
    ))
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module.split(".")[0] in {"trajectron", "environment", "dill", "third_party"}:
                    raise AssertionError(f"Reference import {module} in {path}")
    return [str(path.relative_to(PACKAGE)) for path in paths]


def audit_fold(fold, args):
    store = load_split(fold, "test", args.data_root)
    val = pd.read_csv(args.data_root / fold / "val_manifest.csv")
    cache = CachedNeutralDataset(cache_path(args.cache_root, fold, "test"), store)
    ids = [row.window_id for row in store.manifests]
    truth = cache.arrays["future_metric"].numpy()
    # Use original float64 positions for reproducible Gaussian perturbation.
    observed = np.asarray([
        [store.tracks[row.scene_key][row.target_id][frame][:2]
         for frame in range(row.anchor - 7, row.anchor + 1)]
        for row in store.manifests
    ])
    count = len(ids)
    threshold_path = args.threshold_root / f"{fold}.json"
    threshold = read(threshold_path)
    assert threshold["train_split_only"] is True and threshold["fold"] == fold
    rows = []
    for seed in SEEDS:
        run = args.runs_root / fold / f"seed_{seed}"
        for stage in ("train", "select", "test"):
            stamp = read(run / f"{stage}_verified.json")
            for name, digest in stamp["files"].items():
                assert sha256(run / name) == digest, (run, stage, name)
        training = read(run / "metadata.json")
        config = read(run / "config.json")
        assert training["implementation"] == VERSION and training["epochs"] == 100
        assert training["seed"] == seed and training["fold"] == fold
        assert training["train_cache_used"] and training["val_cache_used"]
        assert training["config_hash"] == digest_json(config)
        assert training["source_fingerprint"] == config["source_fingerprint"] == EXPECTED_SOURCE
        assert config["save_every"] == 10 and config["batch_size"] == 256
        assert len(training["history"]) == 100
        selection = read(run / "checkpoint_selection.json")
        assert selection["implementation"] == VERSION
        assert selection["selection_split"] == "val" and not selection["test_used_for_selection"]
        assert selection["num_samples"] == 20
        candidates = selection["candidates"]
        assert {row["checkpoint_epoch"] for row in candidates} == set(range(10, 101, 10))
        assert all(row["split"] == "val" and row["window_count"] == len(val) for row in candidates)
        selected = min(candidates, key=lambda row: (row["energy_score"], row["mean_ade"], row["checkpoint_epoch"]))
        assert selected == selection["selected"]
        checkpoint = run / selected["checkpoint"]
        checkpoint_digest = sha256(checkpoint)
        summary = read(run / "test_evaluation_summary.json")
        assert summary["implementation"] == VERSION
        assert summary["checkpoint"] == checkpoint.name and summary["num_samples"] == 20
        assert summary["test_conditions"] == [tag for tag, _, _ in CONDITIONS]
        assert len(summary["summaries"]) == len(CONDITIONS)
        for result, (tag, condition, level) in zip(summary["summaries"], CONDITIONS):
            assert result["condition_tag"] == tag and result["window_count"] == count
            assert result["thresholds"] == threshold
            path = run / result["predictions"]
            assert path.is_relative_to(run) and path.name == f"predictions_test_{tag}.npz"
            with np.load(path) as bundle:
                metadata = json.loads(str(bundle["metadata"]))
                assert all(metadata.get(key) == value for key, value in store.identity.items())
                assert metadata["seed"] == seed and metadata["sample_count"] == 20
                assert metadata["condition"] == condition and metadata["level"] == level
                assert metadata["batch_size"] == 128 and metadata["max_windows"] is None
                assert metadata["checkpoint_sha256"] == checkpoint_digest
                assert metadata["checkpoint"] == checkpoint.name
                assert metadata["config_hash"] == training["config_hash"]
                records = [json.loads(str(value)) for value in bundle["records"]]
                assert [row["window_id"] for row in records] == ids
                assert all(
                    row["seed"] == seed and row["sampling_seed"] == seed + index // 128
                    and row["coordinate_unit"] == "meter" and row["dt"] == 0.4
                    for index, row in enumerate(records)
                )
                samples = bundle["samples"]
                assert samples.shape == (count, 20, 12, 2) and np.isfinite(samples).all()
                np.testing.assert_array_equal(bundle["ground_truth"], truth)
                expected = [perturb_history(row, condition, level, corruption_seed(seed, index))
                            for index, row in enumerate(observed)]
                np.testing.assert_array_equal(bundle["observed"], np.asarray([row for row, _ in expected], np.float32))
                np.testing.assert_array_equal(bundle["observed_mask"], np.asarray([mask for _, mask in expected]))
            window_metrics = pd.read_csv(run / result["window_metrics"])
            assert window_metrics["window_id"].tolist() == ids
            assert all(name in window_metrics for name in (
                "min_ade", "min_fde", "mean_ade", "mean_fde", "energy_score", "sample_diversity",
                "speed_violation_rate", "acceleration_violation_rate", "jerk_violation_rate",
            ))
            assert "close_approach_rate" in pd.read_csv(run / result["spatial_consistency"], nrows=1)
            rows.append({
                "model": metadata["model_name"], "protocol": VERSION, "fold": fold,
                "seed": seed, "condition": tag, "window_count": count, "sample_count": 20,
                "prediction_path": str(path.relative_to(args.runs_root.parent)),
                "prediction_sha256": sha256(path), "checkpoint": checkpoint.name,
                "checkpoint_epoch": selected["checkpoint_epoch"], "checkpoint_sha256": checkpoint_digest,
                "config_hash": training["config_hash"], "manifest_hash": store.identity["manifest_hash"],
                "source_fingerprint": training["source_fingerprint"],
            })
    print(f"CAUSAL_FOLD_AUDIT_OK fold={fold} bundles={len(rows)}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=PACKAGE / "outputs/causal_v2/runs")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/causal_v2")
    parser.add_argument("--cache-root", type=Path, default=PROJECT / "data/causal_v2_cache")
    parser.add_argument("--threshold-root", type=Path, default=PACKAGE / "outputs/causal_v2/thresholds")
    args = parser.parse_args()
    audited_sources = check_imports()
    rows = [row for fold in FOLDS for row in audit_fold(fold, args)]
    frame = pd.DataFrame(rows)
    assert len(frame) == 90 and not frame.duplicated(["fold", "seed", "condition"]).any()
    output = args.runs_root.parent
    frame.to_csv(output / "prediction_index.csv", index=False)
    write_json(output / "audit.json", {
        "protocol": VERSION, "status": "passed", "runs": 15, "prediction_bundles": 90,
        "samples_per_window": 20, "test_windows": int(frame.query("seed == 0 and condition == 'clean_0'")["window_count"].sum()),
        "audited_source_paths": audited_sources,
    })
    print("INDEPENDENT_FORMAL_AUDIT_OK runs=15 prediction_bundles=90")


if __name__ == "__main__":
    main()
