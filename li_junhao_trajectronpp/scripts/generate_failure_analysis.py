from __future__ import print_function

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")


def parse_records(bundle):
    records = []
    for value in bundle["records"]:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        records.append(json.loads(str(value)))
    return records


def load_bundle(path):
    with np.load(path) as archive:
        records = parse_records(archive)
        return {
            "observed": archive["observed"],
            "ground_truth": archive["ground_truth"],
            "samples": archive["samples"],
            "observed_mask": archive["observed_mask"] if "observed_mask" in archive else None,
            "by_id": {record["window_id"]: index for index, record in enumerate(records)},
        }


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


def fixed_examples(data_root, fold):
    path = os.path.join(data_root, fold, "test_manifest.csv")
    frame = pd.read_csv(path).sort_values(["scene_name", "scene_t", "target_id"])
    rows = []
    for scene_name, group in frame.groupby("scene_name", sort=False):
        group = group.reset_index(drop=True)
        indices = sorted(set([0, len(group) // 2, len(group) - 1]))
        for rank, index in enumerate(indices):
            row = group.iloc[index]
            rows.append(
                {
                    "fold": fold,
                    "scene_name": scene_name,
                    "window_id": row.window_id,
                    "selection": "fixed_{}".format(rank),
                }
            )
    return pd.DataFrame(rows)


def plot_conditions(fold, fixed, conditions, output):
    count = len(fixed)
    figure, axes = plt.subplots(count, 3, figsize=(12, max(3.3, 2.9 * count)), squeeze=False)
    for row_index, (_, item) in enumerate(fixed.iterrows()):
        window_id = item.window_id
        # One shared scale per example, set from all panels before plotting.
        paths = []
        for _, bundle in conditions:
            index = bundle["by_id"][window_id]
            paths.extend((bundle["observed"][index], bundle["ground_truth"][index]))
            paths.extend(bundle["samples"][index])
        points = np.concatenate(paths)
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = (lower + upper) * 0.5
        radius = max(float((upper - lower).max()) * 0.55, 0.4)
        for column, (label, bundle) in enumerate(conditions):
            axis = axes[row_index][column]
            index = bundle["by_id"][window_id]
            observed = bundle["observed"][index]
            truth = bundle["ground_truth"][index]
            samples = bundle["samples"][index]
            mask = bundle["observed_mask"]
            keep = int(mask[index].sum()) if mask is not None else len(observed)
            axis.plot(observed[:keep, 0], observed[:keep, 1], "ko-", markersize=2, label="Observed")
            if keep < len(observed):
                axis.plot(observed[keep - 1:, 0], observed[keep - 1:, 1], "k--", label="Causal fill")
            axis.plot(truth[:, 0], truth[:, 1], color="#bf4d18", linewidth=1.5, label="Ground truth")
            axis.scatter(*truth[-1], color="#bf4d18", marker="*", s=45, zorder=4)
            for sample in samples:
                axis.plot(sample[:, 0], sample[:, 1], color="#2363a2", alpha=0.25, lw=0.8)
            axis.scatter(samples[:, -1, 0], samples[:, -1, 1], color="#2363a2", alpha=0.6,
                         s=10, label="Sample endpoints")
            axis.set_title(f"{label}  |  {window_id}", fontsize=8)
            axis.set(xlim=(center[0] - radius, center[0] + radius),
                     ylim=(center[1] - radius, center[1] + radius), xlabel="x (m)", ylabel="y (m)")
            axis.set_aspect("equal")
            axis.grid(alpha=0.18)
            if row_index == 0 and column == 0:
                axis.legend(loc="best", fontsize=6)
    figure.suptitle(f"{fold.upper()} target-history perturbations (seed 0)", fontsize=12)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def failure_rows(fold, run_dir):
    metrics_root = os.path.join(run_dir, "evaluation")
    clean = pd.read_csv(os.path.join(metrics_root, "clean_0", "metrics", "window_metrics.csv"))
    noise = pd.read_csv(os.path.join(metrics_root, "noise_020", "metrics", "window_metrics.csv"))
    missing = pd.read_csv(os.path.join(metrics_root, "missing_2", "metrics", "window_metrics.csv"))
    missing = missing.rename(
        columns={
            "mean_ade": "mean_ade_missing2",
            "mean_fde": "mean_fde_missing2",
            "energy_score": "energy_score_missing2",
        }
    )
    joined = clean[[
        "window_id", "scene_name", "mean_ade", "mean_fde", "min_ade", "min_fde",
        "energy_score", "speed_violation_rate", "acceleration_violation_rate",
        "jerk_violation_rate",
    ]].merge(
        noise[["window_id", "mean_ade", "mean_fde", "energy_score"]],
        on="window_id", suffixes=("_clean", "_noise020"), validate="one_to_one"
    ).merge(
        missing[["window_id", "mean_ade_missing2", "mean_fde_missing2", "energy_score_missing2"]],
        on="window_id", validate="one_to_one"
    )
    joined["noise020_mean_ade_degradation"] = joined["mean_ade_noise020"] - joined["mean_ade_clean"]
    joined["missing2_mean_ade_degradation"] = joined["mean_ade_missing2"] - joined["mean_ade_clean"]
    joined["motion_violation_score"] = (
        joined["speed_violation_rate"]
        + joined["acceleration_violation_rate"]
        + joined["jerk_violation_rate"]
    )
    selected = []
    for scene_name, group in joined.groupby("scene_name", sort=False):
        choices = (
            ("clean_top_mean_ade", group.nlargest(1, "mean_ade_clean")),
            ("noise020_degradation", group.nlargest(1, "noise020_mean_ade_degradation")),
            ("missing2_degradation", group.nlargest(1, "missing2_mean_ade_degradation")),
            ("clean_motion_violation", group.sort_values(["motion_violation_score", "jerk_violation_rate", "mean_ade_clean", "window_id"], ascending=[False, False, False, True]).head(1)),
        )
        for case_type, selected_group in choices:
            item = selected_group.iloc[0].to_dict()
            item["fold"] = fold
            item["case_type"] = case_type
            selected.append(item)
    return pd.DataFrame(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "processed"))
    parser.add_argument("--runs-root", default=os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "formal"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs"))
    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    figure_dir = os.path.join(args.runs_root, "figures")
    if not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    fixed_frames = []
    failure_frames = []
    for fold in FOLDS:
        fixed = fixed_examples(args.data_root, fold)
        fixed_frames.append(fixed)
        run_dir = os.path.join(args.runs_root, fold, "seed_0")
        conditions = [
            (label, load_bundle(prediction_path(run_dir, condition)))
            for label, condition in (
                ("Clean", "clean_0"), ("Noise 0.20 m", "noise_020"), ("Missing two", "missing_2")
            )
        ]
        plot_conditions(fold, fixed, conditions, os.path.join(figure_dir, f"{fold}_fixed.png"))
        failures = failure_rows(fold, run_dir)
        failure_frames.append(failures)
        plot_conditions(fold, failures.drop_duplicates("window_id"), conditions,
                        os.path.join(figure_dir, f"{fold}_failures.png"))
        if fold == "eth":
            plot_conditions(fold, fixed.iloc[[len(fixed) // 2]], conditions,
                            os.path.join(figure_dir, "eth_presentation.png"))

    fixed_path = os.path.join(args.output_dir, "formal_fixed_examples.csv")
    failure_path = os.path.join(args.output_dir, "formal_failure_cases.csv")
    pd.concat(fixed_frames, ignore_index=True).to_csv(fixed_path, index=False)
    pd.concat(failure_frames, ignore_index=True).to_csv(failure_path, index=False)
    print("WROTE", fixed_path, failure_path, figure_dir)


if __name__ == "__main__":
    main()
