#!/usr/bin/env python3
"""Generate and evaluate all six test conditions for one independent run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREDICT = os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "scripts", "predict_independent.py")
EVALUATE = os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "scripts", "evaluate.py")


CONDITIONS = (
    ("clean_0", "clean", 0.0, 0),
    ("noise_005", "noise", 0.05, 0),
    ("noise_010", "noise", 0.10, 0),
    ("noise_020", "noise", 0.20, 0),
    ("missing_1", "missing", 1.0, 1),
    ("missing_2", "missing", 2.0, 2),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["eth", "hotel", "univ", "zara1", "zara2"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2"))
    parser.add_argument("--cache-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2_cache"))
    parser.add_argument("--device", default="cuda" if os.path.exists("/dev/nvidia0") else "cpu")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selection_path = os.path.join(args.run_dir, "checkpoint_selection.json")
    if not os.path.isfile(selection_path):
        raise FileNotFoundError("Run checkpoint selection first: {}".format(selection_path))
    with open(selection_path) as handle:
        selection = json.load(handle)
    checkpoint = os.path.join(args.run_dir, selection["selected"]["checkpoint"])
    if selection.get("selection_split") != "val" or selection.get("test_used_for_selection"):
        raise ValueError("Checkpoint selection provenance is not validation-only")

    summaries = []
    for tag, condition, sigma, missing in CONDITIONS:
        output_dir = os.path.join(args.run_dir, "evaluation", tag)
        prediction_path = os.path.join(output_dir, "predictions_test_{}.npz".format(tag))
        metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(output_dir, exist_ok=True)
        if args.force or not os.path.isfile(prediction_path):
            command = [
                sys.executable,
                PREDICT,
                "--fold", args.fold,
                "--seed", str(args.seed),
                "--checkpoint", checkpoint,
                "--split", "test",
                "--condition", condition,
                "--num-samples", str(args.num_samples),
                "--data-root", args.data_root,
                "--cache-root", args.cache_root,
                "--output", prediction_path,
                "--device", args.device,
            ]
            if args.max_windows is not None:
                command.extend(["--max-windows", str(args.max_windows)])
            if condition == "noise":
                command.extend(["--sigma", str(sigma)])
            if condition == "missing":
                command.extend(["--sigma", str(missing)])
            subprocess.check_call(command, cwd=PROJECT_ROOT)
        summary_path = os.path.join(metrics_dir, "summary.json")
        if args.force or not os.path.isfile(summary_path):
            subprocess.check_call(
                [
                    sys.executable,
                    EVALUATE,
                    "--predictions", prediction_path,
                    "--output-dir", metrics_dir,
                    "--thresholds", args.thresholds,
                ],
                cwd=PROJECT_ROOT,
            )
        with open(summary_path) as handle:
            summary = json.load(handle)
        summary["condition_tag"] = tag
        summary["checkpoint"] = os.path.basename(checkpoint)
        for field in ("predictions", "window_metrics", "scene_summary", "spatial_consistency"):
            summary[field] = os.path.relpath(summary[field], args.run_dir)
        summaries.append(summary)

    output = {
        "implementation": "causal_v2",
        "fold": args.fold,
        "seed": args.seed,
        "checkpoint": os.path.basename(checkpoint),
        "selection_split": "val",
        "test_conditions": [item[0] for item in CONDITIONS],
        "num_samples": args.num_samples,
        "summaries": summaries,
    }
    with open(os.path.join(args.run_dir, "test_evaluation_summary.json"), "w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(
        "INDEPENDENT_RUN_EVALUATION_OK fold={} seed={} conditions={} windows={}".format(
            args.fold,
            args.seed,
            len(summaries),
            [item["window_count"] for item in summaries],
        )
    )


if __name__ == "__main__":
    main()
