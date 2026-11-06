"""Shared position-only definitions for the Group 7 trajectory benchmark."""

import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = "causal_v2"
FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")
SEEDS = (0, 1, 2)
DT = 0.4
OBSERVATION_POINTS = 8
PREDICTION_HORIZON = 12
CONDITIONS = (
    ("clean_0", "clean", 0.0),
    ("noise_005", "noise", 0.05),
    ("noise_010", "noise", 0.10),
    ("noise_020", "noise", 0.20),
    ("missing_1", "missing", 1),
    ("missing_2", "missing", 2),
)


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def corruption_seed(training_seed, manifest_index):
    return training_seed * 100003 + manifest_index * 1009


def perturb_history(positions, condition, level, seed):
    """Return filled metric positions and a boolean mask of observed steps."""
    positions = np.array(positions, dtype=np.float64, copy=True)
    mask = np.ones(len(positions), dtype=bool)
    if condition == "noise":
        rng = np.random.RandomState(seed)
        positions += rng.normal(0.0, level, size=positions.shape)
    elif condition == "missing":
        if level not in (1, 2) or len(positions) < level + 2:
            raise ValueError("Missing suffix requires one or two steps and two visible points")
        keep = len(positions) - int(level)
        mask[keep:] = False
        step = positions[keep - 1] - positions[keep - 2]
        for index in range(keep, len(positions)):
            positions[index] = positions[index - 1] + step
    elif condition != "clean":
        raise ValueError(f"Unknown condition {condition}")
    return positions, mask


def history_state(positions):
    """Derive states strictly within the supplied sequence."""
    positions = np.asarray(positions, dtype=np.float64)
    velocity = np.zeros_like(positions)
    acceleration = np.zeros_like(positions)
    velocity[1:] = np.diff(positions, axis=0) / DT
    acceleration[2:] = np.diff(velocity[1:], axis=0) / DT
    return np.concatenate((positions, velocity, acceleration), axis=-1)
