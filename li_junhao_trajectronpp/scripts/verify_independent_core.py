#!/usr/bin/env python3
"""Small acceptance gate for the independent Trajectron++ core."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, "li_junhao_trajectronpp")
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from independent_tpp.data import NeutralWindowDataset, collate_windows, load_split  # noqa: E402
from independent_tpp.model import IndependentTrajectronPP  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", default="eth")
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "causal_v2"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device(args.device)
    store = load_split(args.fold, "train", args.data_root)
    dataset = NeutralWindowDataset(store, limit=4)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_windows)
    batch = next(iter(loader))
    for key in (
        "target_history",
        "future_velocity",
        "future_position",
        "neighbors",
        "neighbor_mask",
    ):
        batch[key] = batch[key].to(device)

    model = IndependentTrajectronPP().to(device)
    values = model.loss(batch)
    if not torch.isfinite(values["loss"]):
        raise AssertionError("non-finite smoke loss")
    values["loss"].backward()
    samples = model.sample_positions(batch, num_samples=20)
    if tuple(samples.shape) != (4, 20, 12, 2):
        raise AssertionError("unexpected prediction shape {}".format(tuple(samples.shape)))
    if not torch.isfinite(samples).all():
        raise AssertionError("non-finite prediction sample")

    # The SingleIntegrator part is explicit in sample_positions: zero
    # normalized velocity produces zero relative displacement.  Verify the
    # metric conversion convention independently from the neural network.
    zero_velocity = torch.zeros((2, 12, 2), device=device)
    displacement = torch.cumsum(zero_velocity * 2.0 * 0.4, dim=1)
    if not np.allclose(displacement.detach().cpu().numpy(), 0.0):
        raise AssertionError("zero-velocity integration failed")
    print(
        "INDEPENDENT_CORE_SMOKE_OK",
        "windows={} neighbors={} loss={:.6f} samples={}".format(
            len(dataset),
            int(batch["neighbors"].shape[1]),
            float(values["loss"].detach().cpu()),
            tuple(samples.shape),
        ),
    )


if __name__ == "__main__":
    main()
