from __future__ import print_function

import argparse
import json
import os

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")
CONDITIONS = ("clean_0", "noise_005", "noise_010", "noise_020", "missing_1", "missing_2")
SAMPLE_BUDGETS = (1, 5, 10, 20)


def prediction_path(run_dir, condition):
    if condition == "clean_0":
        names = ("predictions_test_clean.npz", "predictions_test_clean_0.npz")
    elif condition.startswith("noise_"):
        suffix = condition.split("_")[1]
        names = (
            "predictions_test_noise_{}.npz".format(suffix),
            "predictions_test_noise{}.npz".format(suffix),
        )
    else:
        suffix = condition.split("_")[1]
        names = (
            "predictions_test_missing_{}.npz".format(suffix),
            "predictions_test_missing{}.npz".format(suffix),
        )
    for name in names:
        path = os.path.join(run_dir, "evaluation", condition, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("No prediction bundle for {} under {}".format(condition, run_dir))


def metrics_for_budget_batch(samples, target, budget, batch_size=512):
    """Vectorized U-statistic metrics for a fixed prefix of samples."""
    upper = np.triu_indices(budget, 1)
    values = {key: [] for key in ("min_ade", "min_fde", "mean_ade", "mean_fde", "energy_score", "sample_diversity")}
    for start in range(0, len(samples), batch_size):
        stop = min(start + batch_size, len(samples))
        subset = samples[start:stop, :budget]
        target_batch = target[start:stop]
        difference = subset - target_batch[:, None, :, :]
        distances = np.linalg.norm(difference, axis=-1)
        ade = distances.mean(axis=-1)
        fde = np.linalg.norm(subset[:, :, -1, :] - target_batch[:, None, -1, :], axis=-1)
        sequence_to_target = np.sqrt(np.mean(np.sum(difference ** 2, axis=-1), axis=-1))
        if budget > 1:
            pair_difference = subset[:, :, None, :, :] - subset[:, None, :, :, :]
            pair_distance = np.sqrt(np.mean(np.sum(pair_difference ** 2, axis=-1), axis=-1))
            pair_sum = pair_distance[:, upper[0], upper[1]].sum(axis=1)
            energy = sequence_to_target.mean(axis=1) - pair_sum / float(budget * (budget - 1))
            diversity = pair_sum / float(len(upper[0]))
        else:
            energy = sequence_to_target[:, 0]
            diversity = np.zeros(len(subset), dtype=np.float64)
        values["min_ade"].append(ade.min(axis=1))
        values["min_fde"].append(fde.min(axis=1))
        values["mean_ade"].append(ade.mean(axis=1))
        values["mean_fde"].append(fde.mean(axis=1))
        values["energy_score"].append(energy)
        values["sample_diversity"].append(diversity)
    return {key: np.concatenate(value) for key, value in values.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default=os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "formal"))
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "formal_sample_budget.csv"))
    args = parser.parse_args()
    rows = []
    for fold in FOLDS:
        for seed in (0, 1, 2):
            for condition in CONDITIONS:
                path = prediction_path(
                    os.path.join(args.runs_root, fold, "seed_{}".format(seed)),
                    condition,
                )
                bundle = np.load(path)
                samples = np.asarray(bundle["samples"], dtype=np.float32)
                target = np.asarray(bundle["ground_truth"], dtype=np.float32)
                for budget in SAMPLE_BUDGETS:
                    window_values = metrics_for_budget_batch(samples, target, budget)
                    row = {
                        "fold": fold,
                        "seed": seed,
                        "condition": condition,
                        "sample_budget": budget,
                        "window_count": len(samples),
                    }
                    for column, values in window_values.items():
                        row[column + "_mean"] = float(np.mean(values))
                        row[column + "_std"] = float(np.std(values))
                    rows.append(row)
                bundle.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    overall = frame.groupby(["condition", "sample_budget"]).agg({
        column: ["mean", "std", "count"]
        for column in frame.columns
        if column not in {"fold", "seed", "condition", "sample_budget"}
    }).reset_index()
    overall.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in overall.columns
    ]
    overall_path = args.output.replace(".csv", "_overall.csv")
    overall.to_csv(overall_path, index=False)
    print("WROTE", args.output, overall_path)


if __name__ == "__main__":
    main()
