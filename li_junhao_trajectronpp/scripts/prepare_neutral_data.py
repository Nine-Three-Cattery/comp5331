#!/usr/bin/env python3
"""Build a model-neutral ETH/UCY representation without importing Trajectron++.

This is the clean-room/shared-loader path for the revised experiment.  It reads
the locked raw text release, creates the same five-fold 8->12 manifests, and
writes flat trajectory rows with position/velocity/acceleration.  It does not
import dill, Environment, Scene, Node, or any model repository code.

The existing data/processed/*.pkl files remain reference-baseline artifacts.
The neutral output is intended for the group shared loader and future
independent model implementations.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp"))
from independent_tpp.protocol import VERSION, history_state

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "causal_v2")
RAW_ROOT = os.path.join(
    PROJECT_ROOT, "third_party", "Trajectron-plus-plus",
    "experiments", "pedestrians", "raw", "raw", "all_data"
)

DT = 0.4
FRAME_STRIDE = 10
HISTORY = 7
HORIZON = 12
BOUNDARY_GAP = HISTORY + 1 + HORIZON
TRAIN_RATIO = 0.8

SCENE_FILES = {
    "eth": ["biwi_eth.txt"],
    "hotel": ["biwi_hotel.txt"],
    "univ": ["students001.txt", "students003.txt"],
    "zara1": ["crowds_zara01.txt"],
    "zara2": ["crowds_zara02.txt"],
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source(path):
    data = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["frame_raw", "track_id", "pos_x", "pos_y"],
    )
    data["frame_raw"] = pd.to_numeric(data["frame_raw"])
    data["track_id"] = pd.to_numeric(data["track_id"]).astype(np.int64)
    data["frame_id"] = np.floor(data["frame_raw"] / FRAME_STRIDE).astype(np.int64)
    data["frame_id"] -= int(data["frame_id"].min())
    data.sort_values(["frame_id", "track_id"], inplace=True)
    if data.duplicated(["frame_id", "track_id"]).any():
        raise ValueError("Duplicate frame/track pair in {}".format(path))
    return data.reset_index(drop=True)


def scene_rows(data, scene_id, source_file, split, segment_start, segment_end):
    segment = data[(data["frame_id"] >= segment_start) & (data["frame_id"] <= segment_end)].copy()
    scene_name = "{}__{}__{}".format(scene_id, source_file[:-4], split)
    rows = []
    for track_id in sorted(segment["track_id"].unique()):
        node = segment[segment["track_id"] == track_id].sort_values("frame_id")
        frame_ids = node["frame_id"].to_numpy(dtype=np.int64)
        if len(frame_ids) == 0:
            continue
        if np.any(np.diff(frame_ids) != 1):
            raise ValueError(
                "Track {} in {} has an internal gap in {}".format(
                    track_id, source_file, split
                )
            )
        position = node[["pos_x", "pos_y"]].to_numpy(dtype=np.float64)
        state = history_state(position)
        velocity, acceleration = state[:, 2:4], state[:, 4:6]
        for index, frame_id in enumerate(frame_ids):
            rows.append({
                "scene_id": scene_id,
                "scene_name": scene_name,
                "source_file": source_file[:-4],
                "split": split,
                "frame": int(frame_id - segment_start),
                "target_id": str(int(track_id)),
                "position_x": float(position[index, 0]),
                "position_y": float(position[index, 1]),
                "velocity_x": float(velocity[index, 0]),
                "velocity_y": float(velocity[index, 1]),
                "acceleration_x": float(acceleration[index, 0]),
                "acceleration_y": float(acceleration[index, 1]),
            })
    return rows


def manifest_rows(rows, fold, scene_id, source_file, split, segment_start):
    result = []
    frame = pd.DataFrame(rows)
    for target_id, group in frame.groupby("target_id", sort=True):
        first = int(group["frame"].min())
        last = int(group["frame"].max())
        for anchor in range(first + HISTORY, last - HORIZON + 1):
            absolute_anchor = int(segment_start + anchor)
            result.append({
                "fold": fold,
                "split": split,
                "scene_id": scene_id,
                "scene_name": group["scene_name"].iloc[0],
                "source_file": source_file[:-4],
                "target_id": str(target_id),
                "scene_t": int(anchor),
                "anchor_t": absolute_anchor,
                "obs_start": int(absolute_anchor - HISTORY),
                "obs_end": absolute_anchor,
                "fut_start": int(absolute_anchor + 1),
                "fut_end": int(absolute_anchor + HORIZON),
                "num_observed": int(HISTORY + 1),
                "num_future": int(HORIZON),
                "dt": DT,
                "coordinate_unit": "meter",
                "eligible": True,
                "window_id": "{}:{}:{}:{}".format(
                    scene_id, source_file[:-4], target_id, absolute_anchor
                ),
            })
    return result


def build_fold(fold, output_root, raw_root, expected_hashes):
    fold_root = os.path.join(output_root, fold)
    os.makedirs(fold_root, exist_ok=True)
    all_manifest = []
    raw_hashes = {}
    split_counts = {}

    for scene_id, filenames in SCENE_FILES.items():
        for filename in filenames:
            path = os.path.join(raw_root, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
            raw_hashes[filename] = sha256_file(path)
            if raw_hashes[filename] != expected_hashes[filename]:
                raise ValueError(f"Raw file hash mismatch: {filename}")
            data = read_source(path)
            max_t = int(data["frame_id"].max())
            total = max_t + 1
            if scene_id == fold:
                specs = [("test", 0, max_t)]
            else:
                boundary = int(np.floor(total * TRAIN_RATIO))
                specs = [
                    ("train", 0, boundary - BOUNDARY_GAP - 1),
                    ("val", boundary + BOUNDARY_GAP, max_t),
                ]
            for split, start, end in specs:
                if start < 0 or end <= start:
                    raise ValueError("Invalid {} segment for {}".format(split, filename))
                rows = scene_rows(data, scene_id, filename, split, start, end)
                manifest = manifest_rows(
                    rows, fold, scene_id, filename, split, segment_start=start
                )
                all_manifest.extend(manifest)
                split_counts[split] = split_counts.get(split, 0) + len(manifest)
                rows_path = os.path.join(
                    fold_root, "{}_{}_{}_rows.csv".format(split, scene_id, filename[:-4])
                )
                pd.DataFrame(rows).to_csv(rows_path, index=False)

    manifest_frame = pd.DataFrame(all_manifest).sort_values(
        ["split", "scene_id", "source_file", "target_id", "scene_t"]
    )
    if manifest_frame["window_id"].duplicated().any():
        raise ValueError("Duplicate neutral window ID in {}".format(fold))
    manifest_frame.to_csv(os.path.join(fold_root, "manifest.csv"), index=False)
    for split in ("train", "val", "test"):
        subset = manifest_frame[manifest_frame["split"] == split]
        subset.to_csv(os.path.join(fold_root, "{}_manifest.csv".format(split)), index=False)
    protocol = {
        "fold": fold,
        "dt": DT,
        "frame_stride": FRAME_STRIDE,
        "history_intervals": HISTORY,
        "observation_points": HISTORY + 1,
        "prediction_horizon": HORIZON,
        "boundary_gap": BOUNDARY_GAP,
        "train_ratio": TRAIN_RATIO,
        "scene_files": SCENE_FILES,
        "raw_hashes": raw_hashes,
        "split_counts": split_counts,
        "loader": VERSION,
        "official_model_imports": False,
    }
    with open(os.path.join(fold_root, "protocol.json"), "w") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
    return split_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-root", default=RAW_ROOT)
    parser.add_argument("--raw-hashes", default=os.path.join(
        PROJECT_ROOT, "li_junhao_trajectronpp", "configs", "causal_v2_raw_sha256.json"
    ))
    args = parser.parse_args()
    with open(args.raw_hashes) as handle:
        expected_hashes = json.load(handle)
    for filename, expected_hash in expected_hashes.items():
        if sha256_file(os.path.join(args.raw_root, filename)) != expected_hash:
            raise ValueError(f"Raw file hash mismatch: {filename}")
    os.makedirs(args.output_root, exist_ok=True)
    summary = {}
    for fold in SCENE_FILES:
        summary[fold] = build_fold(fold, args.output_root, args.raw_root, expected_hashes)
    with open(os.path.join(args.output_root, "protocol.json"), "w") as handle:
        json.dump({
            "version": VERSION,
            "dt": DT,
            "observation_points": HISTORY + 1,
            "prediction_horizon": HORIZON,
            "boundary_gap": BOUNDARY_GAP,
            "folds": summary,
            "official_model_imports": False,
        }, handle, indent=2, sort_keys=True)
    print("NEUTRAL_DATA_OK folds={} output={}".format(len(summary), os.path.abspath(args.output_root)))


if __name__ == "__main__":
    main()
