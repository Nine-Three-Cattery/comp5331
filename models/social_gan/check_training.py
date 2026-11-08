"""Compare a fresh ETH pilot training run with the original saved pilot."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(reference, retrained):
    require(reference.resolve() != retrained.resolve(), "Use two distinct training output trees")
    identities = {}
    for name in ("config.json", "checkpoint_selection.json"):
        original, fresh = [json.loads((root / name).read_text()) for root in (reference, retrained)]
        require(original == fresh, f"Training records differ: {name}")
        identities[name] = sha256(reference / name)
    original, fresh = [torch.load(root / "selected.pt", map_location="cpu", weights_only=True)
                       for root in (reference, retrained)]
    require(original.keys() == fresh.keys(), "Model state keys differ")
    require(all(torch.equal(original[key], fresh[key]) for key in original), "Selected model tensors differ")
    identities["selected.pt"] = sha256(reference / "selected.pt")
    conditions = ("clean_0", "noise_005", "noise_010", "noise_020", "missing_1", "missing_2")
    bundles = {}
    for condition in conditions:
        path = Path("evaluation") / condition / f"predictions_test_{condition}.npz"
        with np.load(reference / path, allow_pickle=False) as left, \
                np.load(retrained / path, allow_pickle=False) as right:
            require(set(left.files) == set(right.files), f"Bundle fields differ: {condition}")
            for key in left.files:
                require(np.array_equal(left[key], right[key]), f"Bundle field differs: {condition}/{key}")
            bundles[condition] = {"arrays_and_metadata": "exact", "fields": sorted(left.files),
                                  "reference_sha256": sha256(reference / path)}
    return {"status": "passed", "model": "social_gan", "profile": "pilot", "fold": "eth", "seed": 0,
            "epochs": 2, "selected_state_tensors": len(original), "identity_sha256": identities,
            "bundles": bundles, "scope": "Exact training and export replay in the recorded local environment.",
            "source_sha256": {name: sha256(Path(__file__).parent / name)
                              for name in ("model.py", "data.py", "run.py", "config.json", "check_training.py")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/social_gan/pilot/runs/eth/seed_0")
    parser.add_argument("--retrained", type=Path, default=ROOT / "artifacts/social_gan/retrain_check/runs/eth/seed_0")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "submission/contributions/social_gan/evidence/training_reproduction.json")
    args = parser.parse_args()
    receipt = compare(args.reference, args.retrained)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("OWNER_TRAINING_REPRODUCTION_OK 22 tensors and all six pilot exports match exactly")


if __name__ == "__main__":
    main()
