"""Replay the 15 selected checkpoints on clean validation windows without the training cache."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import torch
from torch.distributions import MultivariateNormal
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from li_junhao_trajectronpp.independent_tpp.data import load_split
from li_junhao_trajectronpp.independent_tpp.protocol import FOLDS, SEEDS, PREDICTION_HORIZON, write_json
from models.social_stgcnn.export import raw_for_batch
from models.social_stgcnn.run import ARTIFACTS, SETTINGS, restore
from models.social_stgcnn.timing import timing_batch


# The trainer evaluates a float32 density and rounds each scene's per-target mean.
# The replay evaluates and sums individual log densities in float64.
NLL_TOLERANCE = 2e-5


def logged_minimum(run, selection, settings):
    with (run / "training.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = settings["formal_epochs"]
    if [int(row["epoch"]) for row in rows] != list(range(1, epochs + 1)):
        raise ValueError(f"Expected all {epochs} training epochs in order")
    if not all(math.isfinite(float(row["train_nll"])) for row in rows):
        raise ValueError("A training epoch lacks a finite loss")

    candidates = [(int(row["epoch"]), float(row["validation_nll"])) for row in rows if row["validation_nll"]]
    scheduled = [e for e in range(1, epochs + 1) if e in (1, epochs) or e % settings["validation_interval"] == 0]
    if [e for e, _ in candidates] != scheduled or not all(math.isfinite(nll) for _, nll in candidates):
        raise ValueError("Validation log differs from the registered candidate schedule")
    saved = [(entry["epoch"], entry["validation_nll"]) for entry in selection["candidates"]]
    if saved != candidates:
        raise ValueError("The checkpoint candidate table differs from the training log")
    return min(candidates, key=lambda item: (item[1], item[0]))


@torch.inference_mode()
def replay_nll(model, store, batch_size):
    total = 0.0
    for start in range(0, len(store.manifests), batch_size):
        rows = store.manifests[start:start + batch_size]
        batch = timing_batch(store, rows)
        raw = raw_for_batch(model, batch, torch.device("cpu")).double()

        # Reconstruct the released label transform independently from the raw
        # neutral target positions, including the last observed position.
        world = np.asarray([[store.state(row.scene_key, row.target_id, row.anchor + offset)[:2]
                             for offset in range(PREDICTION_HORIZON + 1)] for row in rows], dtype=np.float64)
        steps = torch.from_numpy(np.diff(np.around(world, decimals=4), axis=1).astype(np.float32)).double()
        sx, sy = raw[..., 2:4].exp().unbind(-1)
        rho = raw[..., 4].tanh()
        scale = raw.new_zeros((*raw.shape[:-1], 2, 2))
        scale[..., 0, 0] = sx
        scale[..., 1, 0] = sy * rho
        scale[..., 1, 1] = sy / raw[..., 4].cosh()
        density = MultivariateNormal(raw[..., :2], scale_tril=scale)
        total -= density.log_prob(steps).clamp_min(math.log(1e-20)).sum().item()
    value = total / (len(store.manifests) * PREDICTION_HORIZON)
    if not math.isfinite(value):
        raise ValueError("Replayed validation NLL is non-finite")
    return value


def check_run(run, store, fold, seed, settings):
    selection = json.loads((run / "checkpoint_selection.json").read_text())
    best_epoch, best_nll = logged_minimum(run, selection, settings)
    model, config, hashes = restore(run, torch.device("cpu"))
    if (config["fold"], config["seed"], config["stage"], config["epochs"]) != (
            fold, seed, "formal", settings["formal_epochs"]) or config["val_identity"] != store.identity:
        raise ValueError("Run or validation identity does not match the requested replay")
    selected = selection["selected"]
    reported = float(selected["validation_nll"])
    if not math.isfinite(reported):
        raise ValueError("Selected validation NLL is non-finite")
    actual = replay_nll(model, store, settings["batch_size"])
    difference = actual - reported
    tolerance = NLL_TOLERANCE * max(1.0, abs(reported))
    winner_matches = (selected["epoch"], reported) == (best_epoch, best_nll)
    return {
        "passed": winner_matches and abs(difference) <= tolerance,
        "selected_epoch": selected["epoch"], "minimum_candidate_epoch": best_epoch,
        "epochs_checked": settings["formal_epochs"], "candidates_checked": len(selection["candidates"]),
        "validation_windows": len(store.manifests), "validation_manifest_sha256": store.identity["manifest_hash"],
        "checkpoint_sha256": hashes["checkpoint_sha256"], "minimum_candidate_nll": best_nll,
        "recorded_nll": reported, "replayed_nll": actual, "difference": difference, "absolute_tolerance": tolerance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="Formal directory containing fold/seed runs")
    parser.add_argument("--data", type=Path, required=True, help="Neutral causal_v2 data directory")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "selection_check.json", help="Output JSON receipt")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    settings = json.loads(SETTINGS.read_text())
    rows = []
    for fold in FOLDS:
        store = load_split(fold, "val", str(args.data))
        for seed in SEEDS:
            run = args.runs / fold / f"seed_{seed}"
            try:
                result = check_run(run, store, fold, seed, settings)
            except (OSError, KeyError, ValueError, RuntimeError) as error:
                result = {"passed": False, "error": str(error)}
            rows.append({"fold": fold, "seed": seed, **result})
    passed_count = sum(row["passed"] for row in rows)
    passed = passed_count == len(rows)
    summary = {
        "runs_checked": len(rows), "runs_passed": passed_count, "runs_failed": len(rows) - passed_count,
        "max_absolute_difference": max((abs(row["difference"]) for row in rows if "difference" in row), default=None),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, {"passed": passed, "split": "val", "nll_tolerance_factor": NLL_TOLERANCE,
                          "summary": summary, "runs": rows})
    if not passed:
        failed = ", ".join(f"{row['fold']}/{row['seed']}" for row in rows if not row["passed"])
        raise SystemExit(f"Selection replay failed for {failed}. See {args.out}")
    print(f"SELECTION_REPLAY_OK runs={len(rows)} output={args.out}")


if __name__ == "__main__":
    main()
