"""Model-neutral ETH/UCY window reader for the independent T++ core.

The reader is deliberately independent of the official Trajectron++ object
model.  It consumes the flat-row representation produced by
``scripts/prepare_neutral_data.py`` and makes the coordinate choice explicit:

* positions are represented relative to the target's last observed position;
* velocity is divided by 2 m/s and acceleration is divided by 1 m/s^2;
* the decoder predicts normalized velocity and integrates it with dt=0.4;
* predictions are converted back to metric coordinates by the adapter.

The transformation is fitted-free and therefore cannot leak a held-out
scene statistic.  It is recorded as a deliberate deviation from the official
reference preprocessing in the independent-run provenance.
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .protocol import (
    DT, OBSERVATION_POINTS, PREDICTION_HORIZON, VERSION, corruption_seed,
    digest_json, history_state, perturb_history, sha256,
)


HISTORY_INTERVALS = OBSERVATION_POINTS - 1
NEIGHBOR_RADIUS = 3.0
VELOCITY_SCALE = 2.0
ACCELERATION_SCALE = 1.0


@dataclasses.dataclass(frozen=True)
class WindowRecord:
    """One manifest row with stable identity and split metadata."""

    values: Mapping[str, object]

    @property
    def window_id(self) -> str:
        return str(self.values["window_id"])

    @property
    def scene_key(self) -> Tuple[str, str]:
        return str(self.values["scene_id"]), str(self.values["source_file"])

    @property
    def target_id(self) -> str:
        return str(self.values["target_id"])

    @property
    def anchor(self) -> int:
        return int(self.values["scene_t"])


@dataclasses.dataclass
class SceneStore:
    """Dense-enough lookup table for one fold and one split."""

    tracks: MutableMapping[Tuple[str, str], MutableMapping[str, Mapping[int, np.ndarray]]]
    manifests: List[WindowRecord]
    identity: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def state(self, scene_key: Tuple[str, str], target_id: str, frame: int) -> np.ndarray:
        return np.asarray(self.tracks[scene_key][str(target_id)][int(frame)], dtype=np.float64)

    def target_ids(self, scene_key: Tuple[str, str]) -> Iterable[str]:
        return self.tracks.get(scene_key, {}).keys()


def _read_row_file(path: str) -> Tuple[Tuple[str, str], MutableMapping[str, Mapping[int, np.ndarray]]]:
    frame = pd.read_csv(path)
    required = {
        "scene_id", "source_file", "target_id", "frame",
        "position_x", "position_y", "velocity_x", "velocity_y",
        "acceleration_x", "acceleration_y",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("{} is missing {}".format(path, sorted(missing)))
    if frame.duplicated(["target_id", "frame"]).any():
        raise ValueError("Duplicate target/frame rows in {}".format(path))
    scene_id = str(frame["scene_id"].iloc[0]) if len(frame) else ""
    source_file = str(frame["source_file"].iloc[0]) if len(frame) else ""
    tracks: MutableMapping[str, Mapping[int, np.ndarray]] = {}
    for target_id, group in frame.groupby("target_id", sort=True):
        states: Dict[int, np.ndarray] = {}
        for row in group.itertuples(index=False):
            states[int(row.frame)] = np.asarray(
                [
                    row.position_x,
                    row.position_y,
                    row.velocity_x,
                    row.velocity_y,
                    row.acceleration_x,
                    row.acceleration_y,
                ],
                dtype=np.float64,
            )
        tracks[str(target_id)] = states
    return (scene_id, source_file), tracks


def load_split(fold: str, split: str, data_root: str) -> SceneStore:
    """Load one fold/split from the neutral flat-row representation."""

    fold_root = os.path.abspath(os.path.join(data_root, fold))
    manifest_path = os.path.join(fold_root, "{}_manifest.csv".format(split))
    with open(os.path.join(data_root, "protocol.json")) as handle:
        if json.load(handle)["version"] != VERSION:
            raise ValueError(f"Expected {VERSION} data at {data_root}")
    manifest_frame = pd.read_csv(manifest_path)
    manifests = [
        WindowRecord({key: value for key, value in row.items()})
        for row in manifest_frame.to_dict(orient="records")
    ]

    tracks: MutableMapping[Tuple[str, str], MutableMapping[str, Mapping[int, np.ndarray]]] = {}
    pattern = os.path.join(fold_root, "{}_*_rows.csv".format(split))
    row_paths = sorted(glob.glob(pattern))
    if not row_paths:
        raise FileNotFoundError("No neutral row files match {}".format(pattern))
    for path in row_paths:
        scene_key, scene_tracks = _read_row_file(path)
        if scene_key in tracks:
            raise ValueError("Duplicate neutral scene key {}".format(scene_key))
        tracks[scene_key] = scene_tracks

    manifest_keys = {record.scene_key for record in manifests}
    if not manifest_keys.issubset(set(tracks)):
        missing = sorted(manifest_keys.difference(set(tracks)))
        raise ValueError("Manifest scenes have no row file: {}".format(missing))
    identity = {
        "version": VERSION,
        "fold": fold,
        "split": split,
        "manifest_hash": sha256(manifest_path),
        "input_hash": digest_json({os.path.basename(path): sha256(path) for path in row_paths}),
    }
    return SceneStore(tracks=tracks, manifests=manifests, identity=identity)


def _normalized_state(state: np.ndarray, origin: np.ndarray, reference_position: Optional[np.ndarray] = None) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64).copy()
    if reference_position is None:
        reference_position = origin
    state[..., 0:2] = state[..., 0:2] - np.asarray(reference_position)
    state[..., 2:4] = state[..., 2:4] / VELOCITY_SCALE
    state[..., 4:6] = state[..., 4:6] / ACCELERATION_SCALE
    return state


def build_window(
    store: SceneStore,
    record: WindowRecord,
    condition: str = "clean",
    sigma: float = 0.0,
    corruption_seed: int = 0,
) -> Dict[str, object]:
    """Convert one manifest record into target and graph tensors.

    A pedestrian neighbor is retained only when all eight history points are
    present.  Its state is masked at individual times when it lies outside
    the 3m pedestrian interaction radius.  This makes the time-varying graph
    explicit and avoids silently extrapolating neighbor tracks.
    """

    scene_key = record.scene_key
    target_id = record.target_id
    anchor = record.anchor
    history_frames = np.arange(anchor - HISTORY_INTERVALS, anchor + 1, dtype=np.int64)
    future_frames = np.arange(anchor + 1, anchor + PREDICTION_HORIZON + 1, dtype=np.int64)

    target_track = store.tracks[scene_key][target_id]
    observed_clean = np.stack([target_track[int(frame)][:2] for frame in history_frames])
    observed_positions, observed_mask = perturb_history(
        observed_clean, condition, sigma, corruption_seed
    )
    origin = observed_positions[-1].copy()
    target_history = _normalized_state(history_state(observed_positions), origin)

    future_positions = np.stack([target_track[int(frame)][:2] for frame in future_frames])
    label_positions = np.concatenate((observed_clean[-1:], future_positions))
    target_future_velocity = np.diff(label_positions, axis=0) / (DT * VELOCITY_SCALE)
    future_position_relative = future_positions - origin[None, :]

    neighbor_sequences: List[np.ndarray] = []
    neighbor_masks: List[np.ndarray] = []
    for neighbor_id in sorted(store.target_ids(scene_key)):
        if str(neighbor_id) == target_id:
            continue
        neighbor_track = store.tracks[scene_key][str(neighbor_id)]
        if not all(int(frame) in neighbor_track for frame in history_frames):
            continue
        neighbor_positions = np.stack([neighbor_track[int(frame)][:2] for frame in history_frames])
        neighbor_state = history_state(neighbor_positions)
        relative = neighbor_state.copy()
        relative[:, 0:2] = neighbor_state[:, 0:2] - observed_positions
        relative[:, 2:4] = neighbor_state[:, 2:4] / VELOCITY_SCALE
        relative[:, 4:6] = neighbor_state[:, 4:6] / ACCELERATION_SCALE
        distances = np.linalg.norm(relative[:, 0:2], axis=-1)
        active = distances <= NEIGHBOR_RADIUS
        if not np.any(active):
            continue
        relative[~active] = 0.0
        neighbor_sequences.append(relative.astype(np.float32))
        neighbor_masks.append(active.astype(np.float32))

    if neighbor_sequences:
        order = np.argsort(
            [float(np.linalg.norm(sequence[-1, 0:2])) for sequence in neighbor_sequences]
        )
        neighbors = np.stack([neighbor_sequences[index] for index in order], axis=0)
        neighbor_mask = np.stack([neighbor_masks[index] for index in order], axis=0)
    else:
        neighbors = np.zeros((0, OBSERVATION_POINTS, 6), dtype=np.float32)
        neighbor_mask = np.zeros((0, OBSERVATION_POINTS), dtype=np.float32)

    return {
        "window_id": record.window_id,
        "scene_id": str(record.values["scene_id"]),
        "source_file": str(record.values["source_file"]),
        "target_id": target_id,
        "scene_t": anchor,
        "anchor_t": int(record.values["anchor_t"]),
        "dt": DT,
        "condition": condition,
        "sigma": float(sigma),
        "missing": int(sigma) if condition == "missing" else 0,
        "target_history": target_history.astype(np.float32),
        "future_velocity": target_future_velocity.astype(np.float32),
        "future_position": future_position_relative.astype(np.float32),
        "observed_metric": observed_positions.astype(np.float32),
        "observed_mask": observed_mask,
        "future_metric": future_positions.astype(np.float32),
        "origin_metric": origin.astype(np.float32),
        "neighbors": neighbors,
        "neighbor_mask": neighbor_mask,
    }


class NeutralWindowDataset(Dataset):
    """PyTorch dataset over neutral manifest windows."""

    def __init__(
        self,
        store: SceneStore,
        condition: str = "clean",
        sigma: float = 0.0,
        corruption_seed: int = 0,
        limit: Optional[int] = None,
    ):
        self.store = store
        self.condition = condition
        self.sigma = float(sigma)
        self.training_seed = int(corruption_seed)
        self.records = store.manifests if limit is None else store.manifests[: int(limit)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        seed = corruption_seed(self.training_seed, index)
        return build_window(
            self.store,
            record,
            condition=self.condition,
            sigma=self.sigma,
            corruption_seed=seed,
        )


class CachedNeutralDataset(Dataset):
    """Fast clean-window cache generated from the same neutral loader.

    The cache contains no model outputs or learned statistics.  It is only a
    materialized form of NeutralWindowDataset(condition="clean") and is
    therefore safe to reuse across seeds and checkpoints.
    """

    def __init__(self, path: str, store: SceneStore, limit: Optional[int] = None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        if metadata.get("identity") != store.identity or metadata.get("protocol") != protocol_metadata():
            raise ValueError(f"Cache protocol or input mismatch: {path}")
        if [item["window_id"] for item in payload["records"]] != [item.window_id for item in store.manifests]:
            raise ValueError(f"Cache window order mismatch: {path}")
        self.arrays = payload["arrays"]
        self.records = payload["records"]
        if limit is not None:
            length = min(int(limit), len(self.records))
            self.records = self.records[:length]
            self.arrays = {
                key: value[:length] for key, value in self.arrays.items()
            }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = dict(self.records[index])
        item = {
            key: value[index].numpy()
            for key, value in self.arrays.items()
        }
        item.update(record)
        return item


def cache_path(cache_root: str, fold: str, split: str) -> str:
    return os.path.abspath(os.path.join(cache_root, fold, "{}.pt".format(split)))


def collate_windows(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Pad variable-neighbor batches while keeping all metric metadata."""

    max_neighbors = max((item["neighbors"].shape[0] for item in batch), default=0)
    if max_neighbors == 0:
        max_neighbors = 1
    batch_size = len(batch)
    neighbors = np.zeros((batch_size, max_neighbors, OBSERVATION_POINTS, 6), dtype=np.float32)
    neighbor_mask = np.zeros((batch_size, max_neighbors, OBSERVATION_POINTS), dtype=np.float32)
    for index, item in enumerate(batch):
        count = min(item["neighbors"].shape[0], max_neighbors)
        if count:
            neighbors[index, :count] = item["neighbors"][:count]
            neighbor_mask[index, :count] = item["neighbor_mask"][:count]

    return {
        "target_history": torch.from_numpy(np.stack([item["target_history"] for item in batch])),
        "future_velocity": torch.from_numpy(np.stack([item["future_velocity"] for item in batch])),
        "future_position": torch.from_numpy(np.stack([item["future_position"] for item in batch])),
        "observed_metric": torch.from_numpy(np.stack([item["observed_metric"] for item in batch])),
        "observed_mask": torch.from_numpy(np.stack([item["observed_mask"] for item in batch])),
        "future_metric": torch.from_numpy(np.stack([item["future_metric"] for item in batch])),
        "origin_metric": torch.from_numpy(np.stack([item["origin_metric"] for item in batch])),
        "neighbors": torch.from_numpy(neighbors),
        "neighbor_mask": torch.from_numpy(neighbor_mask),
        "records": [
            {
                key: item[key]
                for key in (
                    "window_id", "scene_id", "source_file", "target_id",
                    "scene_t", "anchor_t", "dt", "condition", "sigma", "missing",
                )
            }
            for item in batch
        ],
    }


def dataset_fingerprint(store: SceneStore) -> str:
    """Stable manifest fingerprint for independent-run provenance."""

    digest = hashlib.sha256()
    for record in store.manifests:
        digest.update((record.window_id + "\n").encode("utf-8"))
    return digest.hexdigest()


def protocol_metadata() -> Mapping[str, object]:
    return {
        "loader": VERSION,
        "history_derivative": "window_local_backward_difference",
        "corruption_scope": "target_only",
        "observation_points": OBSERVATION_POINTS,
        "prediction_horizon": PREDICTION_HORIZON,
        "dt": DT,
        "neighbor_radius_m": NEIGHBOR_RADIUS,
        "velocity_scale": VELOCITY_SCALE,
        "acceleration_scale": ACCELERATION_SCALE,
        "position_frame": "target_anchor_relative",
        "neighbor_missing_policy": "exclude_track_if_any_history_point_missing",
        "official_model_imports": False,
    }
