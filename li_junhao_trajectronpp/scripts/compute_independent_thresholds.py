#!/usr/bin/env python3
"""Compute fold-specific diagnostics thresholds from neutral train data only."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp")
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from independent_tpp.data import DT, load_split  # noqa: E402

FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")


def finite_quantile(values, quantile, default):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, quantile)) if values.size else float(default)


def thresholds(store, fold):
    speeds = []
    accelerations = []
    jerks = []
    pair_distances = []
    for scene_tracks in store.tracks.values():
        positions_by_frame = {}
        for states in scene_tracks.values():
            frames = sorted(states)
            values = np.asarray([states[frame] for frame in frames], dtype=np.float64)
            speed = np.linalg.norm(values[1:, 2:4], axis=-1)
            acceleration = np.linalg.norm(values[2:, 4:6], axis=-1)
            jerk_vector = np.diff(values[2:, 4:6], axis=0) / DT
            jerk = np.linalg.norm(jerk_vector, axis=-1)
            speeds.extend(speed[np.isfinite(speed)].tolist())
            accelerations.extend(acceleration[np.isfinite(acceleration)].tolist())
            jerks.extend(jerk[np.isfinite(jerk)].tolist())
            for frame, state in zip(frames, values):
                positions_by_frame.setdefault(frame, []).append(state[0:2])
        for positions in positions_by_frame.values():
            if len(positions) < 2:
                continue
            points = np.asarray(positions, dtype=np.float64)
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
            pair_distances.extend(distances[np.triu_indices(len(points), 1)].tolist())
    return {
        "fold": fold,
        "speed": finite_quantile(speeds, 0.99, 1.0),
        "acceleration": finite_quantile(accelerations, 0.99, 1.0),
        "jerk": finite_quantile(jerks, 0.99, 1.0),
        "close_approach_distance": finite_quantile(pair_distances, 0.05, 0.5),
        "quantiles": {"motion_upper": 0.99, "pair_distance_lower": 0.05},
        "dt": DT,
        "train_split_only": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2"))
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "causal_v2", "thresholds"),
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    all_thresholds = {}
    for fold in FOLDS:
        value = thresholds(load_split(fold, "train", args.data_root), fold)
        all_thresholds[fold] = value
        with open(os.path.join(args.output_dir, "{}.json".format(fold)), "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
    with open(os.path.join(args.output_dir, "all.json"), "w") as handle:
        json.dump(all_thresholds, handle, indent=2, sort_keys=True)
    print("INDEPENDENT_THRESHOLDS_OK folds={}".format(len(all_thresholds)))


if __name__ == "__main__":
    main()
