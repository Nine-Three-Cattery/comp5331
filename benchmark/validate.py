"""Check a model's neutral prediction bundles against the common benchmark."""

import argparse
import csv
import json
from pathlib import Path
import sys
import subprocess

import numpy as np

from contract import FIXTURES, OWNER, ROOT, fingerprint, sha256
from setup import DATA, THRESHOLDS, check_manifests

sys.path.insert(0, str(OWNER))
sys.path.insert(0, str(OWNER / "scripts"))
from independent_tpp.data import load_split
from independent_tpp.protocol import CONDITIONS, corruption_seed, digest_json, perturb_history, write_json
from analyze_sample_budget import metrics_for_budget_batch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positions(store, offsets):
    return np.array([[store.state(row.scene_key, row.target_id, row.anchor + offset)[:2]
                      for offset in offsets] for row in store.manifests])


def verify_bundle(path, store, observed, future, seed, budget, condition, level, run_hashes, result):
    if result is not None:
        require(result["prediction_sha256"] == sha256(path), "prediction hash does not match results.csv")
    with np.load(path, allow_pickle=False) as bundle:
        meta = json.loads(str(bundle["metadata"]))
        expected = {**store.identity, **run_hashes, "seed": seed, "sample_count": budget,
                    "condition": condition, "level": level}
        for key, value in expected.items():
            require(meta[key] == value, f"metadata {key} mismatch")
        fields = ("window_id", "scene_name", "scene_id", "source_file", "scene_t", "anchor_t")
        records = bundle["records"]
        require(records.shape == (len(future),), "unexpected record shape")
        for row, value in zip(store.manifests, records):
            record = json.loads(str(value))
            for key in fields:
                require(record[key] == row.values[key], f"{row.window_id} record {key} mismatch")
            extra = {"target_id": row.target_id, "dt": 0.4, "coordinate_unit": "meter",
                     "seed": seed, "split": "test", "fold": store.identity["fold"],
                     "condition": condition, "sigma": level if condition == "noise" else 0,
                     "missing": level if condition == "missing" else 0, "sample_count": budget}
            require(all(record[key] == value for key, value in extra.items())
                    and type(record["sampling_seed"]) is int, f"{row.window_id} condition or units mismatch")
        samples = bundle["samples"]
        require(samples.shape == (len(future), budget, 12, 2) and np.isfinite(samples).all(),
                "invalid prediction array")
        np.testing.assert_array_equal(bundle["ground_truth"], future)
        histories, masks = zip(*(perturb_history(history, condition, level, corruption_seed(seed, i))
                                 for i, history in enumerate(observed)))
        np.testing.assert_array_equal(bundle["observed"], np.asarray(histories, np.float32))
        require(bundle["observed_mask"].dtype == np.bool_, "observed_mask must be boolean")
        np.testing.assert_array_equal(bundle["observed_mask"], masks)
        metrics = metrics_for_budget_batch(samples.astype(np.float64), future.astype(np.float64), budget)
    if result is not None:
        for name, values in metrics.items():
            np.testing.assert_allclose(float(result[f"{name}_mean"]), values.mean(), rtol=1e-7,
                                       atol=1e-8, err_msg=f"{name} differs from bundle")



def score_bundle(path, output, fold):
    threshold = THRESHOLDS / f"{fold}.json"
    subprocess.run([sys.executable, str(OWNER / "scripts/evaluate.py"),
                    "--predictions", str(path), "--thresholds", str(threshold),
                    "--output-dir", str(output)], check=True, stdout=subprocess.DEVNULL)
    return json.loads((output / "summary.json").read_text())["overall"]


def checkpoint_identity(run, fold, seed, manifest_lock):
    config = json.loads((run / "config.json").read_text())
    selection = json.loads((run / "checkpoint_selection.json").read_text())
    require(selection["selection_split"] == "val" and selection["test_used_for_selection"] is False,
            f"{fold}/{seed} must select on validation only")
    for split in ("train", "val"):
        expected = manifest_lock[fold][split]["sha256"]
        require(config[f"{split}_identity"]["manifest_hash"] == expected,
                f"{fold}/{seed} {split} manifest mismatch")
    checkpoint = selection["selected"]["checkpoint"]
    require(checkpoint == Path(checkpoint).name, "checkpoint must be a filename in the run directory")
    return {"config_hash": digest_json(config), "checkpoint": checkpoint,
            "checkpoint_sha256": sha256(run / checkpoint)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="ID in submission/models.json")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="check all six conditions on ETH seed 0")
    parser.add_argument("--export", action="store_true",
                        help="run the common evaluator and export the formal results.csv")
    args = parser.parse_args()
    if args.pilot and args.export:
        parser.error("--export requires the formal matrix")
    models = {row["id"]: row for row in json.loads((ROOT / "submission/models.json").read_text())}
    require(args.model in models, f"unknown model {args.model}")
    budget = models[args.model]["sample_count"]
    require(budget in (1, 20), "register K=1 or K=20 before the pilot")
    check_manifests()
    lock = json.loads((FIXTURES / "manifests.json").read_text())
    folds = ("eth",) if args.pilot else tuple(lock["folds"])
    seeds = (0,) if args.pilot else (0, 1, 2)
    results_path = ROOT / "submission/contributions" / args.model / "results.csv"
    results = {}
    if not args.pilot and not args.export:
        with results_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        results = {(row["fold"], int(row["seed"]), row["condition"]): row for row in rows}
        expected = {(fold, seed, tag) for fold in folds for seed in seeds for tag, _, _ in CONDITIONS}
        require(len(rows) == len(expected) and results.keys() == expected,
                "results.csv must have all 90 unique fold/seed/condition rows")
    output_root = ROOT / "artifacts" / args.model
    exported = []
    for fold in folds:
        store = load_split(fold, "test", str(DATA))
        observed = positions(store, range(-7, 1))
        future = positions(store, range(1, 13)).astype("float32")
        for seed in seeds:
            run = args.runs / fold / f"seed_{seed}"
            identity = checkpoint_identity(run, fold, seed, lock["folds"])
            for tag, condition, level in CONDITIONS:
                path = run / "evaluation" / tag / f"predictions_test_{tag}.npz"
                result = results.get((fold, seed, tag))
                try:
                    if result is not None:
                        require(result["protocol"] == lock["protocol"] and
                                result["manifest_hash"] == store.identity["manifest_hash"] and
                                int(result["window_count"]) == len(future) and
                                int(result["sample_count"]) == budget and
                                all(result[key] == identity[key] for key in ("config_hash", "checkpoint_sha256")),
                                "results.csv does not identify this run")
                    verify_bundle(path, store, observed, future, seed, budget, condition, level, identity, result)
                    if args.export:
                        score = score_bundle(path, output_root / "metrics" / fold / f"seed_{seed}" / tag, fold)
                        metrics = ("min_ade", "min_fde", "mean_ade", "mean_fde", "energy_score", "sample_diversity")
                        exported.append({
                            "protocol": lock["protocol"], "fold": fold, "seed": seed, "condition": tag,
                            "window_count": len(future), "sample_count": budget,
                            "manifest_hash": store.identity["manifest_hash"],
                            "config_hash": identity["config_hash"], "checkpoint_sha256": identity["checkpoint_sha256"],
                            "prediction_sha256": sha256(path),
                            **{f"{name}_mean": score[f"{name}_mean"] for name in metrics},
                        })
                except (AssertionError, ValueError, KeyError, OSError) as error:
                    raise SystemExit(f"{fold}/{seed}/{tag}: {error}") from error
        print(f"BENCHMARK_FOLD_OK {args.model} {fold}", flush=True)
    if args.export:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = results_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(exported[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(exported)
        temporary.replace(results_path)
    output = output_root / ("pilot_audit.json" if args.pilot else "benchmark_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {
        "model": args.model, "protocol": lock["protocol"], "status": "passed",
        "contract_sha256": fingerprint(), "results_sha256": None if args.pilot else sha256(results_path),
        "prediction_bundles": len(folds) * len(seeds) * len(CONDITIONS),
    })
    print(f"BENCHMARK_AUDIT_OK {output.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from error
