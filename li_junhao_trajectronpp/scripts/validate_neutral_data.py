#!/usr/bin/env python3
"""Validate the model-neutral flat trajectory representation.

This validator deliberately has no dependency on the official Trajectron++
environment classes. It checks the shared manifest, finite state rows, and
every 8-observation/12-future window against the flat rows.
"""

from __future__ import print_function

import argparse
import glob
import os

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "data", "neutral")
FOLDS = ["eth", "hotel", "univ", "zara1", "zara2"]
SPLITS = ["train", "val", "test"]
HISTORY = 7
HORIZON = 12
STATE_COLUMNS = [
    "position_x", "position_y", "velocity_x", "velocity_y",
    "acceleration_x", "acceleration_y",
]


def validate_split(root, fold, split):
    fold_root = os.path.join(root, fold)
    manifest_path = os.path.join(fold_root, "{}_manifest.csv".format(split))
    manifest = pd.read_csv(manifest_path)
    required = {
        "fold", "split", "scene_id", "scene_name", "source_file", "target_id",
        "scene_t", "anchor_t", "obs_start", "obs_end", "fut_start", "fut_end",
        "window_id", "num_observed", "num_future", "dt", "coordinate_unit", "eligible",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise AssertionError("Missing manifest columns {}".format(sorted(missing)))
    if manifest["window_id"].duplicated().any():
        raise AssertionError("Duplicate window IDs in {}".format(manifest_path))
    if not manifest["eligible"].astype(bool).all():
        raise AssertionError("Ineligible row in {}".format(manifest_path))

    row_paths = sorted(glob.glob(os.path.join(fold_root, "{}_*_rows.csv".format(split))))
    if not row_paths:
        raise AssertionError("No neutral row files for {}/{}".format(fold, split))
    rows = pd.concat([pd.read_csv(path) for path in row_paths], ignore_index=True)
    required_rows = {
        "scene_id", "scene_name", "source_file", "split", "frame", "target_id",
    }.union(STATE_COLUMNS)
    missing_rows = required_rows.difference(rows.columns)
    if missing_rows:
        raise AssertionError("Missing row columns {}".format(sorted(missing_rows)))
    if not np.isfinite(rows[STATE_COLUMNS].to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite state in {}/{}".format(fold, split))

    rows["target_id"] = rows["target_id"].astype(str)
    rows["scene_name"] = rows["scene_name"].astype(str)
    row_groups = {
        key: group.set_index("frame")
        for key, group in rows.groupby(["scene_name", "target_id"], sort=False)
    }
    for record in manifest.itertuples(index=False):
        key = (str(record.scene_name), str(record.target_id))
        group = row_groups.get(key)
        if group is None:
            raise AssertionError("Missing target group {} in {}".format(key, manifest_path))
        frames = np.arange(int(record.scene_t) - HISTORY, int(record.scene_t) + HORIZON + 1)
        if not np.isin(frames, group.index.to_numpy()).all():
            raise AssertionError("Incomplete window {} in {}".format(record.window_id, manifest_path))
    return len(manifest), len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_ROOT)
    args = parser.parse_args()
    summary = {}
    for fold in FOLDS:
        summary[fold] = {}
        for split in SPLITS:
            windows, rows = validate_split(args.data_root, fold, split)
            summary[fold][split] = {"windows": windows, "rows": rows}
    print(summary)
    print("NEUTRAL_DATA_VALIDATION_OK")


if __name__ == "__main__":
    main()
