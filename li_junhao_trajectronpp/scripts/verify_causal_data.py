#!/usr/bin/env python3
"""Check all cached tensors against freshly constructed windows and export a handoff fixture."""

import argparse
from pathlib import Path
import sys

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))
from independent_tpp.data import CachedNeutralDataset, NeutralWindowDataset, cache_path, load_split
from independent_tpp.protocol import CONDITIONS, FOLDS, VERSION, corruption_seed, write_json


def positions(store, record, offsets):
    track = store.tracks[record.scene_key][record.target_id]
    return np.stack([track[record.anchor + offset][:2] for offset in offsets])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/causal_v2")
    parser.add_argument("--cache-root", type=Path, default=PROJECT / "data/causal_v2_cache")
    parser.add_argument("--legacy-root", type=Path, default=PROJECT / "data/neutral")
    parser.add_argument("--legacy-cache-root", type=Path, default=PROJECT / "data/neutral_cache")
    parser.add_argument("--output", type=Path, default=PACKAGE / "outputs/causal_v2/corruption_fixture.json")
    args = parser.parse_args()
    fixtures = []
    count = 0
    for fold in FOLDS:
        for split in ("train", "val", "test"):
            manifest = Path(fold) / f"{split}_manifest.csv"
            assert (args.data_root / manifest).read_bytes() == (args.legacy_root / manifest).read_bytes()
            store = load_split(fold, split, args.data_root)
            fresh = NeutralWindowDataset(store)
            cached = CachedNeutralDataset(cache_path(args.cache_root, fold, split), store)
            if fold == "eth" and split == "test":
                try:
                    CachedNeutralDataset(cache_path(args.legacy_cache_root, fold, split), store)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Legacy cache was accepted")
            for index in range(len(fresh)):
                current, saved = fresh[index], cached[index]
                for key in saved:
                    if key in ("neighbors", "neighbor_mask"):
                        end = len(current[key])
                        np.testing.assert_array_equal(current[key], saved[key][:end])
                        assert not np.any(saved[key][end:])
                    else:
                        np.testing.assert_array_equal(current[key], saved[key])
                count += 1
            if split == "test":
                index = len(store.manifests) // 2
                record = store.manifests[index]
                for tag, condition, level in CONDITIONS:
                    dataset = NeutralWindowDataset(store, condition=condition, sigma=level)
                    item = dataset[index]
                    fixtures.append({
                        "fold": fold, "manifest_index": index, "window_id": record.window_id,
                        "manifest_hash": store.identity["manifest_hash"], "condition": tag,
                        "training_seed": 0, "corruption_seed": corruption_seed(0, index),
                        "clean_observed_m": positions(store, record, range(-7, 1)).tolist(),
                        "model_observed_m": item["observed_metric"].tolist(),
                        "observed_mask": item["observed_mask"].tolist(),
                        "future_m": item["future_metric"].tolist(),
                    })
            print(f"CAUSAL_CACHE_EQUAL {fold}/{split} windows={len(fresh)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"protocol": VERSION, "cached_windows_checked": count, "examples": fixtures})
    print(f"CAUSAL_DATA_AUDIT_OK windows={count} fixture_rows={len(fixtures)}")


if __name__ == "__main__":
    main()
