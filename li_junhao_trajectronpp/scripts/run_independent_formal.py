#!/usr/bin/env python3
"""Run the causal formal matrix with verified per-run restart points."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

SCRIPTS = Path(__file__).resolve().parent
PACKAGE = SCRIPTS.parent
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))

from independent_tpp.protocol import CONDITIONS, FOLDS, SEEDS, VERSION, digest_json, sha256, write_json


def run_step(script, args, log):
    with log.open("w") as stream:
        subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                       cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def verified_step(run, stage, signature, paths, script, args, force):
    stamp = run / f"{stage}_verified.json"
    if stamp.exists() and not force:
        saved = json.loads(stamp.read_text())
        if saved.get("signature") == signature and all(
            (run / path).is_file() and sha256(run / path) == digest
            for path, digest in saved["files"].items()
        ) and set(saved["files"]) == {str(path.relative_to(run)) for path in paths}:
            return saved, False
    stamp.unlink(missing_ok=True)
    print(f"CAUSAL_RUN {run.parent.name}/{run.name} {stage}", flush=True)
    run_step(script, args, run / f"{stage}.log")
    saved = {"signature": signature, "files": {
        str(path.relative_to(run)): sha256(path) for path in paths
    }}
    write_json(stamp, saved)
    return saved, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/causal_v2")
    parser.add_argument("--cache-root", type=Path, default=PROJECT / "data/causal_v2_cache")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "outputs/causal_v2/runs")
    parser.add_argument("--threshold-root", type=Path, default=PACKAGE / "outputs/causal_v2/thresholds")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    epochs = sorted(set(range(args.save_every, args.epochs + 1, args.save_every)) | {args.epochs})
    stage_source = {
        "train": [*sorted((PACKAGE / "independent_tpp").glob("*.py")),
                  SCRIPTS / "train_independent.py", SCRIPTS / "predict_independent.py"],
        "select": [SCRIPTS / name for name in ("select_independent_checkpoint.py", "evaluate.py", "metrics.py")],
        "test": [SCRIPTS / name for name in ("evaluate_independent_run.py", "evaluate.py", "metrics.py")],
    }
    source = {stage: digest_json({path.name: sha256(path) for path in paths})
              for stage, paths in stage_source.items()}
    count = 0
    for fold in ((args.fold,) if args.fold else FOLDS):
        fold_data = args.data_root.resolve() / fold
        fold_cache = args.cache_root.resolve() / fold
        split_hashes = {}
        for split in ("train", "val", "test"):
            paths = [fold_data / f"{split}_manifest.csv", *sorted(fold_data.glob(f"{split}_*_rows.csv")),
                     fold_cache / f"{split}.pt"]
            split_hashes[split] = {path.name: sha256(path) for path in paths}
        threshold = args.threshold_root.resolve() / f"{fold}.json"
        threshold_hash = sha256(threshold)
        for seed in ((args.seed,) if args.seed is not None else SEEDS):
            started = time.monotonic()
            run = args.output_dir.resolve() / fold / f"seed_{seed}"
            run.mkdir(parents=True, exist_ok=True)
            common = ["--fold", fold, "--seed", seed, "--device", args.device,
                      "--data-root", args.data_root.resolve(), "--cache-root", args.cache_root.resolve()]
            schedule = {"epochs": args.epochs, "batch_size": args.batch_size, "save_every": args.save_every,
                        "device": args.device, "fold": fold, "seed": seed, "version": VERSION}
            train_signature = digest_json([schedule, source["train"], split_hashes["train"], split_hashes["val"]])
            train_paths = [run / name for name in ("best.pt", "metadata.json", "config.json", "train.log")]
            train_paths += [run / f"checkpoint_epoch_{epoch:03d}.pt" for epoch in epochs]
            train, retrained = verified_step(
                run, "train", train_signature, train_paths, "train_independent.py",
                [*common, "--epochs", args.epochs, "--batch-size", args.batch_size,
                 "--save-every", args.save_every, "--output-dir", args.output_dir.resolve()], args.force,
            )
            select_paths = [run / name for name in ("checkpoint_selection.json", "checkpoint_selection.csv", "select.log")]
            select, reselected = verified_step(
                run, "select", digest_json([train, source["select"]]), select_paths,
                "select_independent_checkpoint.py",
                [*common, "--run-dir", run, "--num-samples", 20, "--force",
                 "--candidates", ",".join(map(str, epochs))], args.force or retrained,
            )
            test_paths = [run / "test_evaluation_summary.json", run / "test.log"]
            for tag, _, _ in CONDITIONS:
                directory = run / "evaluation" / tag
                test_paths.append(directory / f"predictions_test_{tag}.npz")
                test_paths.extend(directory / "metrics" / name for name in (
                    "summary.json", "window_metrics.csv", "scene_summary.csv", "spatial_consistency.csv",
                ))
            _, evaluated = verified_step(
                run, "test", digest_json([select, source["test"], split_hashes["test"], threshold_hash]),
                test_paths, "evaluate_independent_run.py",
                [*common, "--run-dir", run, "--num-samples", 20, "--thresholds", threshold, "--force"],
                args.force or reselected,
            )
            driver = run / "formal_driver_metadata.json"
            if evaluated or reselected or retrained or not driver.is_file():
                write_json(driver, {**schedule, "elapsed_seconds": time.monotonic() - started})
            count += 1
            print(f"CAUSAL_FORMAL_RUN_OK {fold} seed={seed}", flush=True)
    print(f"CAUSAL_FORMAL_MATRIX_OK runs={count}")


if __name__ == "__main__":
    main()
