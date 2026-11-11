#!/usr/bin/env python3
"""Select an independent checkpoint using validation-only stochastic metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREDICT = os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "scripts", "predict_independent.py")
EVALUATE = os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "scripts", "evaluate.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["eth", "hotel", "univ", "zara1", "zara2"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2"))
    parser.add_argument("--cache-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2_cache"))
    parser.add_argument("--device", default="cuda" if os.path.exists("/dev/nvidia0") else "cpu")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint_paths = []
    for name in os.listdir(args.run_dir):
        match = re.fullmatch(r"checkpoint_epoch_([0-9]+)\.pt", name)
        if match:
            checkpoint_paths.append((int(match.group(1)), os.path.join(args.run_dir, name)))
    checkpoint_paths.sort()
    if args.candidates:
        wanted = {int(value) for value in args.candidates.split(",")}
        checkpoint_paths = [item for item in checkpoint_paths if item[0] in wanted]
    if not checkpoint_paths:
        raise ValueError("No independent checkpoint_epoch_*.pt files in {}".format(args.run_dir))

    rows = []
    selection_root = os.path.join(args.run_dir, "selection")
    for epoch, checkpoint in checkpoint_paths:
        candidate_dir = os.path.join(selection_root, "checkpoint_{:03d}".format(epoch))
        prediction_path = os.path.join(candidate_dir, "predictions_val_clean.npz")
        metrics_dir = os.path.join(candidate_dir, "metrics")
        os.makedirs(candidate_dir, exist_ok=True)
        if args.force or not os.path.isfile(prediction_path):
            command = [
                sys.executable,
                PREDICT,
                "--fold", args.fold,
                "--seed", str(args.seed),
                "--checkpoint", checkpoint,
                "--split", "val",
                "--condition", "clean",
                "--num-samples", str(args.num_samples),
                "--data-root", args.data_root,
                "--cache-root", args.cache_root,
                "--output", prediction_path,
                "--device", args.device,
            ]
            if args.max_windows is not None:
                command.extend(["--max-windows", str(args.max_windows)])
            subprocess.check_call(command, cwd=PROJECT_ROOT)
        summary_path = os.path.join(metrics_dir, "summary.json")
        if args.force or not os.path.isfile(summary_path):
            subprocess.check_call(
                [
                    sys.executable,
                    EVALUATE,
                    "--predictions", prediction_path,
                    "--output-dir", metrics_dir,
                ],
                cwd=PROJECT_ROOT,
            )
        with open(summary_path) as handle:
            summary = json.load(handle)
        overall = summary["overall"]
        rows.append(
            {
                "checkpoint_epoch": epoch,
                "checkpoint": os.path.basename(checkpoint),
                "energy_score": overall["energy_score_mean"],
                "mean_ade": overall["mean_ade_mean"],
                "min_ade": overall["min_ade_mean"],
                "mean_fde": overall["mean_fde_mean"],
                "min_fde": overall["min_fde_mean"],
                "window_count": summary["window_count"],
                "split": "val",
            }
        )

    table = pd.DataFrame(rows).sort_values(
        ["energy_score", "mean_ade", "checkpoint_epoch"],
        ascending=[True, True, True],
    )
    selected = table.iloc[0].to_dict()
    output = {
        "implementation": "causal_v2",
        "fold": args.fold,
        "seed": args.seed,
        "criterion": "validation_energy_score_then_mean_ade",
        "selection_split": "val",
        "test_used_for_selection": False,
        "num_samples": args.num_samples,
        "selected": selected,
        "candidates": rows,
    }
    with open(os.path.join(args.run_dir, "checkpoint_selection.json"), "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    table.to_csv(os.path.join(args.run_dir, "checkpoint_selection.csv"), index=False)
    print("INDEPENDENT_CHECKPOINT_SELECTED", json.dumps({
        "fold": args.fold,
        "seed": args.seed,
        "epoch": int(selected["checkpoint_epoch"]),
        "energy_score": selected["energy_score"],
        "candidate_count": len(rows),
        "selection_split": "val",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
