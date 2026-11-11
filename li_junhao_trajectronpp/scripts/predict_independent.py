#!/usr/bin/env python3
"""Generate common-schema prediction bundles from the independent core."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp")
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from independent_tpp.data import (  # noqa: E402
    CachedNeutralDataset,
    NeutralWindowDataset,
    cache_path,
    collate_windows,
    load_split,
)
from independent_tpp.protocol import VERSION, digest_json, sha256
from independent_tpp.model import IndependentTrajectronPP  # noqa: E402


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_checkpoint(path, fold, seed, device, store):
    path = Path(path)
    config = json.loads((path.parent / "config.json").read_text())
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (checkpoint.get("protocol_version") != VERSION
            or checkpoint["config_hash"] != digest_json(config)
            or (checkpoint["fold"], checkpoint["seed"]) != (fold, seed)):
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    if store.identity["split"] == "val" and config["val_identity"] != store.identity:
        raise ValueError("Validation input differs from the training snapshot")
    model = IndependentTrajectronPP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    metadata = {
        **store.identity,
        "model_name": "TrajectronPP-independent-causal-v2",
        "seed": seed,
        "checkpoint": path.name,
        "checkpoint_sha256": sha256(path),
        "config_hash": checkpoint["config_hash"],
        "source_fingerprint": checkpoint["source_fingerprint"],
    }
    return model, metadata


def predict_metric(model, batch, num_samples, device):
    """Prediction interface accepts only history, graph and coordinate origin."""
    features = move_batch({
        key: batch[key] for key in ("target_history", "neighbors", "neighbor_mask")
    }, device)
    origin = batch["origin_metric"].to(device).unsqueeze(1).unsqueeze(2)
    return model.sample_positions(features, num_samples=num_samples) + origin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["eth", "hotel", "univ", "zara1", "zara2"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--run-dir",
        default="",
        help="required only when --checkpoint selected is used",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--condition", choices=["clean", "noise", "missing"], default="clean")
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2"))
    parser.add_argument("--cache-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2_cache"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint == "selected":
        if not args.run_dir:
            raise ValueError("--run-dir is required with --checkpoint selected")
        selection_path = os.path.join(args.run_dir, "checkpoint_selection.json")
        with open(selection_path) as handle:
            selection = json.load(handle)
        if selection.get("selection_split") != "val" or selection.get("test_used_for_selection"):
            raise ValueError("Selected checkpoint is not validation-only")
        args.checkpoint = str(Path(args.run_dir) / selection["selected"]["checkpoint"])

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    store = load_split(args.fold, args.split, args.data_root)
    clean_cache = cache_path(args.cache_root, args.fold, args.split)
    cached = None
    if os.path.isfile(clean_cache):
        cached = CachedNeutralDataset(clean_cache, store, limit=args.max_windows)
    if args.condition == "clean" and cached is not None:
        dataset = cached
    else:
        dataset = NeutralWindowDataset(
            store,
            condition=args.condition,
            sigma=args.sigma,
            corruption_seed=args.seed,
            limit=args.max_windows,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )
    model, metadata = load_checkpoint(args.checkpoint, args.fold, args.seed, device, store)
    metadata.update({
        "split": args.split,
        "condition": args.condition,
        "level": args.sigma,
        "sample_count": args.num_samples,
        "batch_size": args.batch_size,
        "max_windows": args.max_windows,
    })

    all_samples = []
    all_truth = []
    all_observed = []
    all_masks = []
    all_records = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            torch.manual_seed(args.seed + batch_index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + batch_index)
            metric_samples = predict_metric(model, batch, args.num_samples, device)
            all_samples.append(metric_samples.cpu().numpy())
            all_truth.append(batch["future_metric"].numpy())
            all_observed.append(batch["observed_metric"].numpy())
            all_masks.append(batch["observed_mask"].numpy().astype(bool))
            for record in batch["records"]:
                output_record = dict(record)
                output_record.update({
                    "fold": args.fold,
                    "split": args.split,
                    "scene_name": "{}__{}__{}".format(
                        record["scene_id"], record["source_file"], args.split
                    ),
                    "seed": args.seed,
                    "condition": args.condition,
                    "sigma": float(args.sigma) if args.condition == "noise" else 0.0,
                    "missing": int(args.sigma) if args.condition == "missing" else 0,
                    "sample_count": args.num_samples,
                    "sampling_seed": args.seed + batch_index,
                    "coordinate_unit": "meter",
                    "model_name": "TrajectronPP-independent-causal-v2",
                })
                all_records.append(json.dumps(output_record, sort_keys=True))

    samples = np.concatenate(all_samples, axis=0) if all_samples else np.zeros((0, args.num_samples, 12, 2))
    ground_truth = np.concatenate(all_truth, axis=0) if all_truth else np.zeros((0, 12, 2))
    observed = np.concatenate(all_observed, axis=0) if all_observed else np.zeros((0, 8, 2))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        samples=samples,
        ground_truth=ground_truth,
        observed=observed,
        records=np.asarray(all_records),
        observed_mask=np.concatenate(all_masks),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print("INDEPENDENT_PREDICTION_OK", json.dumps({
        "output": os.path.abspath(args.output),
        "windows": int(samples.shape[0]),
        "samples": int(samples.shape[1]) if samples.ndim == 4 else 0,
        "shape": list(samples.shape),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
