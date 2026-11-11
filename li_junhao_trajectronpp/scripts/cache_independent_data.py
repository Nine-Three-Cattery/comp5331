#!/usr/bin/env python3
"""Materialize clean neutral windows for efficient independent training."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp")
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from independent_tpp.data import (  # noqa: E402
    DT,
    NEIGHBOR_RADIUS,
    NeutralWindowDataset,
    cache_path,
    dataset_fingerprint,
    load_split,
    protocol_metadata,
)


ARRAY_KEYS = (
    "target_history",
    "future_velocity",
    "future_position",
    "observed_metric",
    "observed_mask",
    "future_metric",
    "origin_metric",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        choices=["eth", "hotel", "univ", "zara1", "zara2"],
        default="",
        help="cache one fold; omit to cache every fold",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="")
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2"))
    parser.add_argument("--cache-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2_cache"))
    args = parser.parse_args()

    folds = [args.fold] if args.fold else ["eth", "hotel", "univ", "zara1", "zara2"]
    splits = [args.split] if args.split else ["train", "val", "test"]
    for fold in folds:
        for split in splits:
            store = load_split(fold, split, args.data_root)
            dataset = NeutralWindowDataset(store, condition="clean")
            items = [dataset[index] for index in range(len(dataset))]
            max_neighbors = max(
                (item["neighbors"].shape[0] for item in items),
                default=0,
            )
            max_neighbors = max(1, max_neighbors)
            arrays = {}
            for key in ARRAY_KEYS:
                arrays[key] = torch.from_numpy(
                    np.stack([item[key] for item in items]).astype(np.float32)
                )
            neighbors = np.zeros(
                (len(items), max_neighbors, 8, 6), dtype=np.float32
            )
            neighbor_mask = np.zeros(
                (len(items), max_neighbors, 8), dtype=np.float32
            )
            records = []
            for index, item in enumerate(items):
                count = min(item["neighbors"].shape[0], max_neighbors)
                if count:
                    neighbors[index, :count] = item["neighbors"][:count]
                    neighbor_mask[index, :count] = item["neighbor_mask"][:count]
                records.append(
                    {
                        key: item[key]
                        for key in (
                            "window_id",
                            "scene_id",
                            "source_file",
                            "target_id",
                            "scene_t",
                            "anchor_t",
                            "dt",
                            "condition",
                            "sigma",
                            "missing",
                        )
                    }
                )
            arrays["neighbors"] = torch.from_numpy(neighbors)
            arrays["neighbor_mask"] = torch.from_numpy(neighbor_mask)
            output_path = cache_path(args.cache_root, fold, split)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(
                {
                    "arrays": arrays,
                    "records": records,
                    "metadata": {
                        "fold": fold,
                        "split": split,
                        "window_count": len(items),
                        "max_neighbors": max_neighbors,
                        "manifest_fingerprint": dataset_fingerprint(store),
                        "identity": store.identity,
                        "protocol": protocol_metadata(),
                        "dt": DT,
                        "neighbor_radius_m": NEIGHBOR_RADIUS,
                        "official_model_imports": False,
                    },
                },
                output_path,
            )
            with open(output_path.replace(".pt", ".json"), "w") as handle:
                json.dump(
                    {
                        "fold": fold,
                        "split": split,
                        "window_count": len(items),
                        "max_neighbors": max_neighbors,
                        "manifest_fingerprint": dataset_fingerprint(store),
                        "identity": store.identity,
                        "protocol": protocol_metadata(),
                        "cache_path": os.path.abspath(output_path),
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            print(
                "INDEPENDENT_CACHE_OK fold={} split={} windows={} max_neighbors={}".format(
                    fold, split, len(items), max_neighbors
                )
            )


if __name__ == "__main__":
    main()
