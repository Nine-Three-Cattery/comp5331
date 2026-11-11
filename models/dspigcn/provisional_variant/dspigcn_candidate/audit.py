"""Audit saved predictions against manifests and recompute all accuracy scores."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from .data import SCENES, DataConfig, digest, load_fold
from .evaluation import CONDITIONS, metrics, motion, perturb, training_thresholds
from .runner import save_json, load_checkpoint


def audit_run(directory, expected_samples=20, data_root=None, expected_scene=None, expected_seed=None):
    directory = Path(directory)
    evaluation = directory / "evaluation"
    manifests = {}
    for split in ("train", "val", "test"):
        path = (evaluation if split == "test" else directory) / f"manifest_{split}.json"
        manifest = json.loads(path.read_text())
        expected = manifest.pop("manifest_sha256")
        if digest(manifest) != expected or digest(manifest["sample_ids"]) != manifest["sample_ids_sha256"]:
            raise ValueError(f"Corrupted manifest {path}")
        manifest["manifest_sha256"] = expected
        manifests[split] = manifest
    train, val, test = (manifests[s] for s in ("train", "val", "test"))
    model, anchors, state = load_checkpoint(directory / "best.pt", "cpu")
    run = json.loads((directory / "run.json").read_text())
    if expected_scene is not None and state["scene"] != expected_scene:
        raise ValueError("Checkpoint is from the wrong held-out scene")
    if expected_seed is not None and state["seed"] != expected_seed:
        raise ValueError("Checkpoint is from the wrong training seed")
    if (state["scene"], state["seed"]) != (run["scene"], run["seed"]) or state["scene"] != test["held_out"]:
        raise ValueError("Checkpoint, run metadata and test scene do not match")
    if state["data_config"] != test["config"] or state["model_config"] != run["config"]["model"]:
        raise ValueError("Checkpoint configuration differs from saved run")
    if not anchors.fitted.item() or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Invalid model/anchor checkpoint")
    for split in ("train", "val"):
        if state[f"{split}_manifest_sha256"] != manifests[split]["manifest_sha256"]:
            raise ValueError("Checkpoint data identity differs from manifest")
        if set(manifests[split]["files_sha256"]) & set(test["files_sha256"]):
            raise ValueError("Held-out scene appears in training/validation data")
    with (directory / "history.csv").open() as f:
        history = list(csv.DictReader(f))
    best = min(history, key=lambda row: float(row["val_mean_trajectory_ADE"]))
    if int(best["epoch"]) != state["epoch"] or len(history) != run.get("completed_epochs"):
        raise ValueError("Checkpoint selection or completed training history is inconsistent")
    if not np.isclose(float(best["val_mean_trajectory_ADE"]), state["validation_ADE"]):
        raise ValueError("Checkpoint validation score differs from training history")
    for filename, segment in train["segments"].items():
        if segment["last"] >= val["segments"][filename]["first"]:
            raise ValueError("Training and validation segments overlap")
    test_set = None
    if data_root is not None:
        for split in ("train", "val", "test"):
            dataset = load_fold(data_root, state["scene"], split, DataConfig(**state["data_config"]))
            if dataset.manifest != manifests[split]:
                raise ValueError(f"{split} manifest does not match the current source release")
            if split == "test":
                test_set = dataset
            if split == "train":
                thresholds = training_thresholds(dataset, state["data_config"]["dt"])
                if thresholds != state["thresholds"]:
                    raise ValueError("Motion thresholds do not match training-only estimation")
    evaluation_settings = json.loads((evaluation / "evaluation.json").read_text())
    with (evaluation / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    if {row["condition"] for row in rows} != set(CONDITIONS) or len(rows) != len(CONDITIONS):
        raise ValueError("Missing or duplicated robustness conditions")
    for row in rows:
        if row["scene"] != state["scene"] or int(row["seed"]) != state["seed"]:
            raise ValueError("Results contain a different scene or seed")
        if row["test_set_id"] != test["manifest_sha256"] or int(row["sample_count"]) != expected_samples:
            raise ValueError("Result identity or sample budget mismatch")
        ids, recomputed = [], []
        close, pairs = 0, 0
        with zipfile.ZipFile(evaluation / row["condition"] / "predictions.zip") as archive:
            if len(archive.namelist()) != test["windows"]:
                raise ValueError("Missing prediction windows")
            for index, name in enumerate(sorted(archive.namelist())):
                with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as arrays:
                    pred, target = arrays["predictions"], arrays["target"]
                    if pred.shape[0] != expected_samples or pred.shape[1] != test["config"]["pred_len"]:
                        raise ValueError("Incorrect prediction dimensions")
                    if len(arrays["sample_ids"]) != pred.shape[2]:
                        raise ValueError("Prediction/identity count mismatch")
                    sigma, missing = CONDITIONS[row["condition"]]
                    expected_observations = perturb(arrays["clean_observations"], arrays["sample_ids"],
                                                    sigma, missing, evaluation_settings["corruption_seed"])
                    if not np.array_equal(expected_observations, arrays["observations"]):
                        raise ValueError("Saved observations do not match the stated corruption")
                    if test_set is not None:
                        example = test_set[index]
                        for key, target_key in (("target", "future"), ("clean_observations", "observations"),
                                                ("frames", "frames"), ("pedestrians", "pedestrians")):
                            if not np.array_equal(arrays[key], example[target_key]):
                                raise ValueError(f"Saved {key} does not match original source data")
                    ids.extend(arrays["sample_ids"].tolist())
                    diagnostics, pair_counts = motion(pred, arrays["observations"], state["data_config"]["dt"], state["thresholds"])
                    recomputed.append({**metrics(pred, target), **diagnostics})
                    close += pair_counts["close_pairs"]
                    pairs += pair_counts["pair_timesteps"]
        if ids != test["sample_ids"] or len(ids) != int(row["trajectories"]):
            raise ValueError("Evaluated population differs from test manifest")
        with (evaluation / row["condition"] / "per_trajectory.csv").open() as f:
            per_trajectory = list(csv.DictReader(f))
        if [r["sample_id"] for r in per_trajectory] != ids:
            raise ValueError("Per-trajectory table population differs from exports")
        for metric in recomputed[0]:
            individual = np.concatenate([part[metric] for part in recomputed])
            score = individual.mean()
            if not np.allclose(individual, [float(r[metric]) for r in per_trajectory], atol=1e-6, rtol=1e-6):
                raise ValueError(f"Per-trajectory metric mismatch: {metric}")
            if not np.isclose(score, float(row[metric]), atol=1e-6, rtol=1e-6):
                raise ValueError(f"Metric mismatch: {row['condition']} {metric}")
        if pairs != int(row["pair_timesteps"]) or not np.isclose(close / pairs if pairs else 0, float(row["close_approach_rate"])):
            raise ValueError("Close-approach count or aggregation mismatch")
    checkpoint = directory / "best.pt"
    return {"directory": str(directory), "scene": test["held_out"], "seed": state["seed"], "conditions": len(rows),
            "windows": test["windows"], "trajectories": test["trajectories"],
            "source_data_verified": data_root is not None,
            "test_set_id": test["manifest_sha256"],
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--matrix", action="store_true", help="Require all 5 scenes x 3 seeds")
    parser.add_argument("--data-root", help="Rebuild manifests and verify targets/corruptions against raw files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    if args.matrix and args.data_root is None:
        parser.error("--matrix requires --data-root for source-data verification")
    dirs = [root / scene / f"seed_{seed}" for scene in SCENES for seed in range(3)] if args.matrix else [root]
    reports = []
    combined_rows = []
    runtime_setting = None
    for directory in dirs:
        report = audit_run(directory, data_root=args.data_root,
                           expected_scene=directory.parent.name if args.matrix else None,
                           expected_seed=int(directory.name.split("_")[-1]) if args.matrix else None)
        reports.append(report)
        print(json.dumps(report), flush=True)
        if args.matrix:
            with (directory / "evaluation/results.csv").open() as f:
                combined_rows.extend(csv.DictReader(f))
            runtime = json.loads((directory / "runtime.json").read_text())
            if (runtime["pedestrians"], runtime["samples"], runtime["warmups"], runtime["repeats"]) != (32, 20, 10, 100):
                raise ValueError(f"Incompatible runtime measurement budget: {directory}")
            setting = (runtime["device"], runtime["environment"], runtime["boundaries"])
            if runtime_setting is not None and setting != runtime_setting:
                raise ValueError("Runtime settings/hardware differ between runs")
            runtime_setting = setting
            for field in ("preprocessing_ms", "prediction_ms", "sampling_ms"):
                if not all(np.isfinite(v) and v >= 0 for v in runtime[field].values()):
                    raise ValueError("Invalid runtime measurements")
            from PIL import Image
            for condition in CONDITIONS:
                folder = directory / "evaluation" / condition
                failures = json.loads((folder / "failures.json").read_text())
                if len(failures) != 3:
                    raise ValueError("Missing failure examples")
                figures = [folder / f"example_{i}.png" for i in range(3)]
                figures += [folder / f"failure_{entry['window_index']}.png" for entry in failures]
                for figure in figures:
                    with Image.open(figure) as image:
                        image.verify()
    if args.matrix:
        with (root / "results.csv").open() as f:
            if list(csv.DictReader(f)) != combined_rows:
                raise ValueError("Combined results table differs from the audited run tables")
    save_json(args.output, {"passed": True, "runs": len(reports), "conditions": sum(r["conditions"] for r in reports),
                            "checks": "checkpoint/config/validation selection, populations, source data when supplied, corruptions, all accuracy/motion metrics and per-trajectory tables",
                            "matrix_checks": "combined CSV, matching runtime budgets/hardware, all example/failure figures" if args.matrix else None,
                            "details": reports, "team_protocol_verified": False})


if __name__ == "__main__":
    main()
