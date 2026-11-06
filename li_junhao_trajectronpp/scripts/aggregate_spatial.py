from __future__ import print_function

import argparse
import os

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        default=os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "formal"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "formal_spatial_results.csv"
        ),
    )
    args = parser.parse_args()

    rows = []
    for fold in ("eth", "hotel", "univ", "zara1", "zara2"):
        fold_root = os.path.join(args.runs_root, fold)
        if not os.path.isdir(fold_root):
            continue
        for seed_name in sorted(os.listdir(fold_root)):
            if not seed_name.startswith("seed_"):
                continue
            seed = int(seed_name.split("_")[1])
            eval_root = os.path.join(fold_root, seed_name, "evaluation")
            if not os.path.isdir(eval_root):
                continue
            for condition in sorted(os.listdir(eval_root)):
                path = os.path.join(
                    eval_root, condition, "metrics", "spatial_consistency.csv"
                )
                if not os.path.isfile(path):
                    continue
                frame = pd.read_csv(path)
                if frame.empty:
                    continue
                numeric = [
                    "target_count",
                    "min_distance_mean",
                    "min_distance_p05",
                    "min_distance_p95",
                    "close_approach_rate",
                ]
                row = {"fold": fold, "seed": seed, "condition": condition}
                row["row_count"] = len(frame)
                for column in numeric:
                    row[column + "_mean"] = float(frame[column].mean())
                    row[column + "_std"] = float(frame[column].std(ddof=0))
                rows.append(row)

    if not rows:
        raise ValueError("No spatial_consistency.csv files found")
    result = pd.DataFrame(rows).sort_values(["fold", "seed", "condition"])
    result.to_csv(args.output, index=False)
    by_fold = (
        result.groupby(["fold", "condition"])
        .agg({
            column: ["mean", "std", "count"]
            for column in result.columns
            if column not in {"fold", "seed", "condition"}
        })
        .reset_index()
    )
    by_fold.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in by_fold.columns
    ]
    by_fold_path = args.output.replace(".csv", "_by_fold.csv")
    by_fold.to_csv(by_fold_path, index=False)
    overall = (
        result.groupby("condition")
        .agg({
            column: ["mean", "std", "count"]
            for column in result.columns
            if column not in {"fold", "seed", "condition"}
        })
        .reset_index()
    )
    overall.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in overall.columns
    ]
    overall_path = args.output.replace(".csv", "_overall.csv")
    overall.to_csv(overall_path, index=False)
    print("WROTE", args.output, by_fold_path, overall_path, "rows", len(result))


if __name__ == "__main__":
    main()
