#!/usr/bin/env python3
"""Model-neutral strata and heading diagnostics for independent formal runs."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp")
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from independent_tpp.data import DT, NEIGHBOR_RADIUS, load_split  # noqa: E402

FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")
SEEDS = (0, 1, 2)
CONDITIONS = ("clean_0", "noise_005", "noise_010", "noise_020", "missing_1", "missing_2")
METRICS = (
    "min_ade",
    "min_fde",
    "mean_ade",
    "mean_fde",
    "energy_score",
    "sample_diversity",
    "speed_violation_rate",
    "acceleration_violation_rate",
    "jerk_violation_rate",
)


def wrap(values):
    return np.arctan2(np.sin(values), np.cos(values))


def prediction_path(run_dir, condition):
    suffix = condition.split("_", 1)[1] if "_" in condition else ""
    if condition == "clean_0":
        names = ("predictions_test_clean.npz", "predictions_test_clean_0.npz")
    elif condition.startswith("noise_"):
        names = (
            "predictions_test_noise_{}.npz".format(suffix),
            "predictions_test_noise{}.npz".format(suffix),
        )
    else:
        names = (
            "predictions_test_missing_{}.npz".format(suffix),
            "predictions_test_missing{}.npz".format(suffix),
        )
    for name in names:
        path = os.path.join(run_dir, "evaluation", condition, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("No prediction bundle for {} under {}".format(condition, run_dir))


def training_thresholds(store):
    turns = []
    densities = []
    for scene_tracks in store.tracks.values():
        by_frame = {}
        for states in scene_tracks.values():
            frames = sorted(states)
            velocity = np.asarray([states[frame][2:4] for frame in frames], dtype=np.float64)
            speed = np.linalg.norm(velocity, axis=-1)
            heading = np.arctan2(velocity[:, 1], velocity[:, 0])
            changes = np.abs(wrap(heading[1:] - heading[:-1]))
            valid = (speed[1:] > 0.05) & (speed[:-1] > 0.05) & np.isfinite(changes)
            turns.extend(changes[valid].tolist())
            for frame, state in zip(frames, [states[value] for value in frames]):
                by_frame.setdefault(frame, []).append(state[0:2])
        for positions in by_frame.values():
            if len(positions) < 2:
                continue
            points = np.asarray(positions, dtype=np.float64)
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
            densities.extend(
                np.sum((distances <= NEIGHBOR_RADIUS) & (distances > 0), axis=1).astype(int).tolist()
            )
    return {
        "turn_threshold_rad": float(np.quantile(turns, 0.95)) if turns else float(np.pi / 4.0),
        "density_threshold_neighbors_3m": float(np.quantile(densities, 0.50)) if densities else 0.0,
        "turn_quantile": 0.95,
        "density_quantile": 0.50,
        "attention_radius": NEIGHBOR_RADIUS,
        "dt": DT,
    }


def load_records(bundle):
    return [json.loads(value.decode() if isinstance(value, bytes) else str(value)) for value in bundle["records"]]


def make_labels(store, bundle, records, thresholds):
    observed = np.asarray(bundle["observed"], dtype=np.float64)
    truth = np.asarray(bundle["ground_truth"], dtype=np.float64)
    rows = []
    for index, record in enumerate(records):
        full = np.concatenate([observed[index, -1:, :], truth[index]], axis=0)
        velocity = np.diff(full, axis=0) / DT
        speed = np.linalg.norm(velocity, axis=-1)
        heading = np.arctan2(velocity[:, 1], velocity[:, 0])
        changes = np.abs(wrap(heading[1:] - heading[:-1]))
        valid = (speed[1:] > 0.05) & (speed[:-1] > 0.05) & np.isfinite(changes)
        max_turn = float(np.max(changes[valid])) if np.any(valid) else 0.0
        scene_key = (str(record["scene_id"]), str(record["source_file"]))
        anchor = int(record["scene_t"])
        target = store.state(scene_key, str(record["target_id"]), anchor)[0:2]
        neighbor_count = 0
        for target_id, states in store.tracks[scene_key].items():
            if str(target_id) == str(record["target_id"]) or anchor not in states:
                continue
            if np.linalg.norm(states[anchor][0:2] - target) <= NEIGHBOR_RADIUS:
                neighbor_count += 1
        rows.append(
            {
                "window_id": record["window_id"],
                "fold": record["fold"],
                "scene_name": record["scene_name"],
                "seed": int(record.get("seed", 0)),
                "turn_score_rad": max_turn,
                "turning": max_turn >= thresholds["turn_threshold_rad"],
                "neighbor_count_3m": neighbor_count,
                "dense": neighbor_count >= thresholds["density_threshold_neighbors_3m"],
                "turn_threshold_rad": thresholds["turn_threshold_rad"],
                "density_threshold_neighbors_3m": thresholds["density_threshold_neighbors_3m"],
            }
        )
    return pd.DataFrame(rows)


def heading_rows(bundle, records, threshold, fold, seed, condition):
    samples = np.asarray(bundle["samples"], dtype=np.float64)
    observed = np.asarray(bundle["observed"], dtype=np.float64)
    full = np.concatenate(
        [
            np.repeat(observed[:, -1:, None, :], samples.shape[1], axis=2),
            np.transpose(samples, (0, 2, 1, 3)),
        ],
        axis=1,
    )
    velocity = np.diff(full, axis=1) / DT
    speed = np.linalg.norm(velocity, axis=-1)
    heading = np.arctan2(velocity[..., 1], velocity[..., 0])
    changes = np.abs(wrap(heading[:, 1:, :] - heading[:, :-1, :]))
    valid = (speed[:, 1:, :] > 0.05) & (speed[:, :-1, :] > 0.05)
    values = np.where(valid, changes, np.nan)
    has_valid = np.isfinite(values).any(axis=(1, 2))
    max_turn = np.zeros(len(samples), dtype=np.float64)
    mean_turn = np.zeros(len(samples), dtype=np.float64)
    p95_turn = np.zeros(len(samples), dtype=np.float64)
    if np.any(has_valid):
        valid_values = values[has_valid]
        max_turn[has_valid] = np.nanmax(valid_values, axis=(1, 2))
        mean_turn[has_valid] = np.nanmean(valid_values, axis=(1, 2))
        p95_turn[has_valid] = np.nanpercentile(valid_values, 95, axis=(1, 2))
    return {
        "fold": fold,
        "seed": seed,
        "condition": condition,
        "window_count": len(samples),
        "turn_threshold_rad": threshold,
        "turn_rate_mean_mean": float(np.mean(mean_turn)),
        "turn_rate_mean_std": float(np.std(mean_turn)),
        "turn_rate_max_mean": float(np.mean(max_turn)),
        "turn_rate_max_std": float(np.std(max_turn)),
        "turn_rate_max_p95_mean": float(np.mean(p95_turn)),
        "turn_violation_rate_mean": float(np.mean(max_turn >= threshold)),
        "turn_violation_rate_std": float(np.std(max_turn >= threshold)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default=os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "independent_formal"))
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "neutral"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "independent_formal_stats"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    label_frames = []
    thresholds_output = {}
    heading_rows_output = []
    for fold in FOLDS:
        train_store = load_split(fold, "train", args.data_root)
        test_store = load_split(fold, "test", args.data_root)
        thresholds = training_thresholds(train_store)
        thresholds_output[fold] = thresholds
        for seed in SEEDS:
            run_dir = os.path.join(args.runs_root, fold, "seed_{}".format(seed))
            clean_bundle = np.load(prediction_path(run_dir, "clean_0"))
            clean_records = load_records(clean_bundle)
            labels = make_labels(test_store, clean_bundle, clean_records, thresholds)
            labels["seed"] = seed
            label_frames.append(labels)
            clean_bundle.close()
            for condition in CONDITIONS:
                bundle = np.load(prediction_path(run_dir, condition))
                records = load_records(bundle)
                heading_rows_output.append(
                    heading_rows(bundle, records, thresholds["turn_threshold_rad"], fold, seed, condition)
                )
                bundle.close()

    labels = pd.concat(label_frames, ignore_index=True)
    labels_path = os.path.join(args.output_dir, "independent_formal_strata_labels.csv")
    labels.to_csv(labels_path, index=False)
    metric_rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            run_root = os.path.join(args.runs_root, fold, "seed_{}".format(seed), "evaluation")
            run_labels = labels[(labels.fold == fold) & (labels.seed == seed)]
            for condition in CONDITIONS:
                frame = pd.read_csv(os.path.join(run_root, condition, "metrics", "window_metrics.csv"))
                frame = frame.merge(run_labels, on="window_id", how="inner", validate="one_to_one")
                for dimension, column in (("turn", "turning"), ("density", "dense")):
                    for value, group in frame.groupby(column, dropna=False):
                        for metric in METRICS:
                            if metric not in group.columns:
                                continue
                            metric_rows.append(
                                {
                                    "fold": fold,
                                    "seed": seed,
                                    "condition": condition,
                                    "dimension": dimension,
                                    "stratum": (
                                        "turning" if dimension == "turn" and bool(value)
                                        else "straight" if dimension == "turn"
                                        else "dense" if bool(value) else "sparse"
                                    ),
                                    "metric": metric,
                                    "mean": float(group[metric].mean()),
                                    "std": float(group[metric].std(ddof=0)),
                                    "count": int(len(group)),
                                }
                            )
    result = pd.DataFrame(metric_rows)
    result_path = os.path.join(args.output_dir, "independent_formal_strata_results.csv")
    result.to_csv(result_path, index=False)
    summary = (
        result.groupby(["condition", "dimension", "stratum", "metric"])
        .agg(mean=("mean", "mean"), std=("mean", "std"), count=("mean", "count"))
        .reset_index()
    )
    summary_path = os.path.join(args.output_dir, "independent_formal_strata_summary.csv")
    summary.to_csv(summary_path, index=False)
    thresholds_path = os.path.join(args.output_dir, "independent_formal_strata_thresholds.json")
    with open(thresholds_path, "w") as handle:
        json.dump(thresholds_output, handle, indent=2, sort_keys=True)

    heading = pd.DataFrame(heading_rows_output)
    heading_path = os.path.join(args.output_dir, "independent_formal_heading_results.csv")
    heading.to_csv(heading_path, index=False)
    heading_overall = heading.groupby("condition").agg(
        {
            column: ["mean", "std", "count"]
            for column in heading.columns
            if column not in {"fold", "seed", "condition"}
        }
    ).reset_index()
    heading_overall.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in heading_overall.columns
    ]
    heading_overall_path = os.path.join(args.output_dir, "independent_formal_heading_results_overall.csv")
    heading_overall.to_csv(heading_overall_path, index=False)
    print(
        "INDEPENDENT_DIAGNOSTICS_OK strata_rows={} heading_rows={}".format(
            len(result), len(heading)
        )
    )


if __name__ == "__main__":
    main()
