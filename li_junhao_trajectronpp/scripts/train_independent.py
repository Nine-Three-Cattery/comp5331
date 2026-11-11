#!/usr/bin/env python3
"""Train the clean-room Trajectron++ pedestrian core.

The script intentionally imports only ``independent_tpp`` and the neutral
loader.  It is suitable for the ETH smoke/pilot first; full formal runs should
be launched with a separate output root for every fold and seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

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
    dataset_fingerprint,
    load_split,
    protocol_metadata,
)
from independent_tpp.protocol import VERSION, digest_json, write_json
from independent_tpp.model import IndependentTrajectronPP  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in ("protocol.py", "data.py", "model.py", "train_independent.py", "predict_independent.py"):
        path = os.path.join(PACKAGE_ROOT, "independent_tpp", name)
        if not os.path.isfile(path):
            path = os.path.join(PACKAGE_ROOT, "scripts", name)
        if os.path.isfile(path):
            digest.update(name.encode("utf-8"))
            with open(path, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()


def evaluate_loss(model, loader, device, limit_batches=None):
    model.eval()
    totals = {"loss": 0.0, "reconstruction_nll": 0.0, "kl": 0.0}
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            values = model.loss(move_batch(batch, device), temperature=0.5)
            for key in totals:
                totals[key] += float(values[key].detach().cpu())
            count += 1
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


def sigmoid_anneal(step, start, finish, center_step, steps_lo_to_hi):
    """Smoothly increase a loss weight using the paper-style schedule."""

    if steps_lo_to_hi <= 0:
        return float(finish)
    slope = 4.0 / float(steps_lo_to_hi)
    value = 1.0 / (1.0 + math.exp(-slope * (float(step) - float(center_step))))
    return float(start + (finish - start) * value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=["eth", "hotel", "univ", "zara1", "zara2"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2"))
    parser.add_argument("--cache-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2_cache"))
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp", "outputs", "independent_runs"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--kl-weight", type=float, default=100.0)
    parser.add_argument("--kl-min", type=float, default=0.07)
    parser.add_argument("--kl-weight-start", type=float, default=0.0)
    parser.add_argument("--kl-crossover", type=float, default=400.0)
    parser.add_argument("--kl-sigmoid-divisor", type=float, default=4.0)
    parser.add_argument("--tau-init", type=float, default=2.0)
    parser.add_argument("--tau-final", type=float, default=0.05)
    parser.add_argument("--tau-decay-rate", type=float, default=0.997)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = os.path.abspath(os.path.join(args.output_dir, args.fold, "seed_{}".format(args.seed)))
    os.makedirs(run_dir, exist_ok=True)

    train_store = load_split(args.fold, "train", args.data_root)
    val_store = load_split(args.fold, "val", args.data_root)
    train_cache = cache_path(args.cache_root, args.fold, "train")
    val_cache = cache_path(args.cache_root, args.fold, "val")
    if os.path.isfile(train_cache):
        train_dataset = CachedNeutralDataset(train_cache, train_store, limit=args.max_train_windows)
        train_cache_used = True
    else:
        train_dataset = NeutralWindowDataset(train_store, limit=args.max_train_windows)
        train_cache_used = False
    if os.path.isfile(val_cache):
        val_dataset = CachedNeutralDataset(val_cache, val_store, limit=args.max_val_windows)
        val_cache_used = True
    else:
        val_dataset = NeutralWindowDataset(val_store, limit=args.max_val_windows)
        val_cache_used = False
    config = {
        key: value for key, value in vars(args).items()
        if key not in ("data_root", "cache_root", "output_dir")
    }
    config.update({
        "protocol": protocol_metadata(),
        "source_fingerprint": source_fingerprint(),
        "train_identity": train_store.identity,
        "val_identity": val_store.identity,
    })
    config_hash = digest_json(config)
    write_json(os.path.join(run_dir, "config.json"), config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
    )

    model = IndependentTrajectronPP(
        kl_weight=args.kl_weight,
        kl_min=args.kl_min,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val = float("inf")
    best_epoch = None
    history = []
    checkpoint_config = {
        "latent_classes": model.latent_classes, "horizon": model.horizon,
        "dt": model.dt, "kl_weight": args.kl_weight, "kl_min": args.kl_min,
    }
    started = time.time()

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_totals = {"loss": 0.0, "reconstruction_nll": 0.0, "kl": 0.0}
        train_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            temperature = max(
                args.tau_final,
                args.tau_init * (args.tau_decay_rate ** global_step),
            )
            kl_weight = sigmoid_anneal(
                global_step,
                args.kl_weight_start,
                args.kl_weight,
                args.kl_crossover,
                args.kl_crossover / args.kl_sigmoid_divisor,
            )
            model.kl_weight = kl_weight
            values = model.loss(
                move_batch(batch, device),
                temperature=temperature,
            )
            values["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            for key in train_totals:
                train_totals[key] += float(values[key].detach().cpu())
            train_count += 1
            global_step += 1
        train_metrics = {key: value / max(train_count, 1) for key, value in train_totals.items()}
        model.kl_weight = args.kl_weight
        val_metrics = evaluate_loss(model, val_loader, device)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "temperature": temperature,
            "kl_weight": kl_weight,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": checkpoint_config,
            "protocol_version": VERSION,
            "config_hash": config_hash,
            "source_fingerprint": config["source_fingerprint"],
            "fold": args.fold,
            "seed": args.seed,
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
        }
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(checkpoint, os.path.join(run_dir, f"checkpoint_epoch_{epoch:03d}.pt"))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save(checkpoint, os.path.join(run_dir, "best.pt"))

    metadata = {
        "implementation": VERSION,
        "config_hash": config_hash,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "fold": args.fold,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "save_every": args.save_every,
        "kl_weight": args.kl_weight,
        "kl_min": args.kl_min,
        "kl_weight_start": args.kl_weight_start,
        "kl_crossover": args.kl_crossover,
        "kl_sigmoid_divisor": args.kl_sigmoid_divisor,
        "tau_init": args.tau_init,
        "tau_final": args.tau_final,
        "tau_decay_rate": args.tau_decay_rate,
        "max_train_windows": args.max_train_windows,
        "max_val_windows": args.max_val_windows,
        "data_root": os.path.abspath(args.data_root),
        "cache_root": os.path.abspath(args.cache_root),
        "train_cache_used": train_cache_used,
        "val_cache_used": val_cache_used,
        "device": str(device),
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "train_manifest_fingerprint": dataset_fingerprint(train_store),
        "val_manifest_fingerprint": dataset_fingerprint(val_store),
        "protocol": protocol_metadata(),
        "source_fingerprint": source_fingerprint(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "global_steps": global_step,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "official_model_imports": False,
    }
    write_json(os.path.join(run_dir, "metadata.json"), metadata)
    print("INDEPENDENT_TRAIN_OK", json.dumps({
        "run_dir": run_dir,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
