#!/usr/bin/env python3
"""Measure the Group 7 clean 128-window benchmark on the selected checkpoints."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import pandas as pd
import torch

PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))
from independent_tpp.data import NeutralWindowDataset, collate_windows, load_split
from independent_tpp.protocol import FOLDS, SEEDS, write_json
from predict_independent import load_checkpoint, predict_metric


def benchmark(store, model, device):
    dataset = NeutralWindowDataset(store, limit=128)
    if len(dataset) != 128:
        raise ValueError("Runtime benchmark requires 128 target windows")
    measurements = []

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for repetition in range(13):
            torch.manual_seed(repetition)
            synchronize()
            start = time.perf_counter()
            batches = []
            for offset in range(0, len(dataset), 32):
                batch = collate_windows([dataset[index] for index in range(offset, offset + 32)])
                batches.append({key: batch[key].to(device) for key in (
                    "target_history", "neighbors", "neighbor_mask", "origin_metric",
                )})
            synchronize()
            ready = time.perf_counter()
            predictions = [predict_metric(model, batch, 20, device) for batch in batches]
            synchronize()
            predicted = time.perf_counter()
            for prediction in predictions:
                prediction.cpu()
            synchronize()
            end = time.perf_counter()
            if repetition >= 3:
                measurements.append({
                    "preprocessing_ms": (ready - start) * 1000,
                    "prediction_sampling_ms": (predicted - ready) * 1000,
                    "end_to_end_ms": (end - start) * 1000,
                })
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/causal_v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=pci.device_id,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip() if device.type == "cuda" else ""
    except (OSError, subprocess.CalledProcessError):
        driver = ""
    rows = []
    repetitions = []
    for fold in FOLDS:
        store = load_split(fold, "test", args.data_root)
        for seed in SEEDS:
            run = args.runs_root / fold / f"seed_{seed}"
            selection = json.loads((run / "checkpoint_selection.json").read_text())
            checkpoint = run / selection["selected"]["checkpoint"]
            model, metadata = load_checkpoint(checkpoint, fold, seed, device, store)
            samples = benchmark(store, model, device)
            row = {"fold": fold, "seed": seed, "checkpoint": checkpoint.name,
                   "checkpoint_sha256": metadata["checkpoint_sha256"], "windows": 128,
                   "samples": 20, "batch_size": 32, "warmups": 3, "measurements": 10,
                   "device": str(device), "hardware": hardware}
            row.update({key: statistics.median(sample[key] for sample in samples)
                        for key in samples[0]})
            rows.append(row)
            repetitions.append({"fold": fold, "seed": seed, "measurements": samples})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    write_json(args.output.with_suffix(".json"), {
        "hardware": hardware, "driver_and_pci_device": driver,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "boundary": ("In-memory neutral positions to CPU metric samples. Preprocessing includes "
                     "causal graph/window construction, collation and device transfer. Inference "
                     "includes sampling and integration. End-to-end also includes CPU readback. "
                     "Disk I/O and one-time model/data loading are excluded."),
        "window_ids_by_fold": {
            fold: pd.read_csv(args.data_root / fold / "test_manifest.csv")["window_id"].head(128).tolist()
            for fold in FOLDS
        },
        "repetitions": repetitions,
    })
    print(f"CAUSAL_RUNTIME_OK rows={len(rows)}")


if __name__ == "__main__":
    main()
