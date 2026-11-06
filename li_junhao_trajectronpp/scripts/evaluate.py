from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from metrics import add_threshold_flags, compute_window_metrics


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_thresholds(path):
    if not path:
        return {}
    with open(path, "r") as handle:
        return json.load(handle)


def spatial_rows(samples, records, threshold):
    frame = pd.DataFrame(records)
    rows = []
    group_columns = ["scene_name", "scene_t", "condition", "sigma", "missing"]
    for key, group in frame.groupby(group_columns):
        indices = group.index.to_numpy()
        if len(indices) < 2:
            continue
        trajectories = samples[indices]
        differences = trajectories[:, None, :, :, :] - trajectories[None, :, :, :, :]
        distances = np.linalg.norm(differences, axis=-1)
        upper = np.triu_indices(len(indices), 1)
        pair_distances = distances[upper[0], upper[1]]
        min_per_sample = pair_distances.min(axis=(0, 2))
        row = {
            "scene_name": key[0],
            "scene_t": int(key[1]),
            "condition": key[2],
            "sigma": float(key[3]),
            "missing": int(key[4]),
            "target_count": int(len(indices)),
            "min_distance_mean": float(np.mean(min_per_sample)),
            "min_distance_p05": float(np.percentile(min_per_sample, 5)),
            "min_distance_p95": float(np.percentile(min_per_sample, 95)),
        }
        if threshold is not None:
            row["close_approach_rate"] = float(
                np.mean(min_per_sample < float(threshold))
            )
        rows.append(row)
    return rows


def flatten_record(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default="")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    bundle = np.load(args.predictions)
    samples = np.asarray(bundle["samples"], dtype=np.float64)
    ground_truth = np.asarray(bundle["ground_truth"], dtype=np.float64)
    observed = np.asarray(bundle["observed"], dtype=np.float64)
    records = [flatten_record(value) for value in bundle["records"]]
    if samples.ndim != 4 or samples.shape[-1] != 2:
        raise ValueError("Expected samples [N,K,T,2], got {}".format(samples.shape))
    if ground_truth.shape != samples.shape[:1] + samples.shape[2:]:
        raise ValueError(
            "Ground truth shape {} incompatible with {}".format(
                ground_truth.shape, samples.shape
            )
        )
    if observed.shape[0] != samples.shape[0]:
        raise ValueError("Observed and samples count differ")
    if len(records) != samples.shape[0]:
        raise ValueError("Records and samples count differ")

    thresholds = load_thresholds(args.thresholds)
    rows = []
    for index, record in enumerate(records):
        metric = compute_window_metrics(
            samples[index],
            ground_truth[index],
            observed[index, -1],
            float(record.get("dt", 0.4)),
        )
        add_threshold_flags(metric, thresholds)
        metric.update(
            {
                "fold": record.get("fold"),
                "split": record.get("split"),
                "scene_id": record.get("scene_id"),
                "scene_name": record.get("scene_name"),
                "source_file": record.get("source_file"),
                "target_id": record.get("target_id"),
                "scene_t": record.get("scene_t"),
                "anchor_t": record.get("anchor_t"),
                "window_id": record.get("window_id"),
                "condition": record.get("condition"),
                "sigma": record.get("sigma", 0.0),
                "missing": record.get("missing", 0),
            }
        )
        rows.append(metric)

    frame = pd.DataFrame(rows)
    window_path = os.path.join(args.output_dir, "window_metrics.csv")
    frame.to_csv(window_path, index=False)

    numeric_columns = [
        column for column in frame.columns
        if column not in {
            "fold", "split", "scene_id", "scene_name", "source_file",
            "target_id", "window_id", "condition", "sigma", "missing"
        } and pd.api.types.is_numeric_dtype(frame[column])
    ]
    grouped = frame.groupby(
        ["fold", "split", "scene_id", "condition", "sigma", "missing"]
    )
    summary = grouped[numeric_columns].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary_path = os.path.join(args.output_dir, "scene_summary.csv")
    summary.to_csv(summary_path, index=False)

    spatial = spatial_rows(
        samples,
        records,
        thresholds.get("close_approach_distance"),
    )
    spatial_path = os.path.join(args.output_dir, "spatial_consistency.csv")
    pd.DataFrame(spatial).to_csv(spatial_path, index=False)

    # The registered protocol reports an equal-weight scene macro average.
    # Keep the pooled window average as an audit field, because overlapping
    # windows are useful for diagnostics but must not dominate the headline
    # result merely because one scene has more valid anchors.
    scene_means = (
        frame.groupby(
            ["fold", "split", "scene_id", "condition", "sigma", "missing"],
            dropna=False,
        )[numeric_columns]
        .mean()
        .reset_index()
    )
    overall = {}
    pooled = {}
    for column in numeric_columns:
        overall[column + "_mean"] = float(scene_means[column].mean())
        overall[column + "_std"] = float(scene_means[column].std(ddof=0))
        pooled[column + "_mean"] = float(frame[column].mean())
        pooled[column + "_std"] = float(frame[column].std(ddof=0))
    metadata = {
        "predictions": os.path.abspath(args.predictions),
        "window_count": int(len(frame)),
        "sample_count": int(samples.shape[1]),
        "horizon": int(samples.shape[2]),
        "thresholds": thresholds,
        "window_metrics": window_path,
        "scene_summary": summary_path,
        "spatial_consistency": spatial_path,
        "aggregation": "scene_macro_mean_of_window_means",
        "overall": overall,
        "overall_window_pooled": pooled,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
