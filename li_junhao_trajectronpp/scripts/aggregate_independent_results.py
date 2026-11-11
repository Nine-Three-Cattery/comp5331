#!/usr/bin/env python3
"""Aggregate independent run summaries without mixing reference artifacts."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")
CONDITIONS = ("clean_0", "noise_005", "noise_010", "noise_020", "missing_1", "missing_2")


def condition_row(tag):
    if tag not in CONDITIONS:
        raise ValueError("Unexpected independent condition {}".format(tag))
    return tag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        default=os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "causal_v2", "runs"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "causal_v2", "results", "results.csv"),
    )
    args = parser.parse_args()

    rows = []
    for fold in FOLDS:
        fold_root = os.path.join(args.runs_root, fold)
        if not os.path.isdir(fold_root):
            continue
        for seed_name in sorted(os.listdir(fold_root)):
            run_dir = os.path.join(fold_root, seed_name)
            summary_path = os.path.join(run_dir, "test_evaluation_summary.json")
            if not os.path.isfile(summary_path):
                continue
            with open(summary_path) as handle:
                payload = json.load(handle)
            if payload.get("implementation") != "causal_v2":
                raise ValueError(f"Unexpected experiment at {summary_path}")
            seed = int(seed_name.split("_")[1])
            for summary in payload["summaries"]:
                overall = summary["overall"]
                row = {
                    "fold": fold,
                    "seed": seed,
                    "condition": condition_row(summary["condition_tag"]),
                    "window_count": summary["window_count"],
                    "sample_count": summary["sample_count"],
                    "checkpoint": payload["checkpoint"],
                }
                for key, value in overall.items():
                    row[key] = value
                rows.append(row)
    if not rows:
        raise ValueError("No independent test summaries found under {}".format(args.runs_root))
    frame = pd.DataFrame(rows)
    expected = {(fold, seed, condition) for fold in FOLDS for seed in range(3)
                for condition in CONDITIONS}
    actual = set(frame[["fold", "seed", "condition"]].itertuples(index=False, name=None))
    if actual != expected or len(frame) != len(expected):
        raise ValueError("Expected the complete causal five-fold, three-seed, six-condition matrix")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    frame.to_csv(args.output, index=False)
    numeric = [
        column
        for column in frame.columns
        if column not in {"fold", "seed", "condition", "checkpoint"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    fold_summary = frame.groupby(["fold", "condition"])[numeric].agg(["mean", "std", "count"]).reset_index()
    fold_summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in fold_summary.columns
    ]
    fold_path = args.output.replace(".csv", "_by_fold.csv")
    fold_summary.to_csv(fold_path, index=False)
    overall = frame.groupby(["condition"])[numeric].agg(["mean", "std", "count"]).reset_index()
    overall.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in overall.columns
    ]
    overall_path = args.output.replace(".csv", "_overall.csv")
    overall.to_csv(overall_path, index=False)
    print(
        "INDEPENDENT_AGGREGATION_OK runs={} conditions={} folds={}".format(
            len(frame[["fold", "seed"]].drop_duplicates()),
            len(frame["condition"].unique()),
            len(frame["fold"].unique()),
        )
    )


if __name__ == "__main__":
    main()
