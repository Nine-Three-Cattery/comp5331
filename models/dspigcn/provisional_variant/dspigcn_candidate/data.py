"""ETH/UCY preparation with explicit splits, identities and source hashes."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SCENES = {"eth": ["biwi_eth.txt"], "hotel": ["biwi_hotel.txt"],
          "univ": ["students001.txt", "students003.txt"],
          "zara1": ["crowds_zara01.txt"], "zara2": ["crowds_zara02.txt"]}


@dataclass(frozen=True)
class DataConfig:
    obs_len: int = 8
    pred_len: int = 12
    dt: float = 0.4
    stride: int = 1
    train_fraction: float = 0.8
    gap: int = 19


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_tracks(path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] not in (4, 6) or not np.isfinite(data).all():
        raise ValueError(f"Expected finite frame, pedestrian, x, y [vx, vy] columns: {path}")
    # The DSPIGCN artifact appends velocities to four of the scene files. These
    # are deliberately ignored: all model states derive from observed positions.
    data = data[:, :4]
    if not np.equal(data[:, :2], np.floor(data[:, :2])).all():
        raise ValueError(f"Frame and pedestrian identifiers must be integers: {path}")
    if len(np.unique(data[:, :2], axis=0)) != len(data):
        raise ValueError(f"Duplicate (frame, pedestrian) row: {path}")
    return data[np.lexsort((data[:, 1], data[:, 0]))]


def windows(data, source, frames, cfg):
    length = cfg.obs_len + cfg.pred_len
    all_frames = np.unique(data[:, 0]).astype(np.int64)
    increments, counts = np.unique(np.diff(all_frames), return_counts=True)
    if not len(increments):
        return []
    frame_step = increments[counts.argmax()]
    by_frame = {int(f): {int(p): xy for _, p, *xy in data[data[:, 0] == f]} for f in frames}
    result = []
    for start in range(0, len(frames) - length + 1, cfg.stride):
        frame_ids = frames[start:start + length]
        if not np.all(np.diff(frame_ids) == frame_step):
            continue
        people = sorted(set.intersection(*(set(by_frame[int(f)]) for f in frame_ids)))
        if not people:
            continue
        trajectory = np.array([[by_frame[int(f)][p] for p in people] for f in frame_ids], dtype=np.float32)
        result.append({"observations": trajectory[:cfg.obs_len], "future": trajectory[cfg.obs_len:],
                       "ids": [f"{source}:{int(frame_ids[0])}:{p}" for p in people],
                       "scene_id": f"{source}:{int(frame_ids[0])}", "frames": frame_ids,
                       "pedestrians": np.array(people), "source": source})
    return result


class TrajectoryDataset(Dataset):
    def __init__(self, examples, manifest):
        self.examples, self.manifest = examples, manifest
        if not examples:
            raise ValueError("No eligible trajectories; check source files and split configuration")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def load_fold(data_root, held_out, split, cfg=DataConfig()):
    if held_out not in SCENES or split not in {"train", "val", "test"}:
        raise ValueError("Unknown scene or split")
    if cfg.stride < 1 or not 0 < cfg.train_fraction < 1 or cfg.gap < cfg.obs_len + cfg.pred_len - 1:
        raise ValueError("Invalid stride/fraction or insufficient split boundary gap")
    examples, files, segments = [], {}, {}
    for scene, names in SCENES.items():
        if (scene == held_out) != (split == "test"):
            continue
        for name in names:
            root = Path(data_root)
            candidates = [root / name, root / scene / "test" / name]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Missing {name}; expected {candidates}")
            raw = path.read_bytes()
            files[name] = hashlib.sha256(raw).hexdigest()
            data = read_tracks(path)
            frames = np.unique(data[:, 0]).astype(np.int64)
            if split != "test":
                boundary = int(len(frames) * cfg.train_fraction)
                frames = frames[:boundary] if split == "train" else frames[boundary + cfg.gap:]
            segments[name] = {"first": int(frames[0]) if len(frames) else None,
                              "last": int(frames[-1]) if len(frames) else None,
                              "frame_count": len(frames)}
            examples.extend(windows(data, name, frames, cfg))
    identities = [identity for ex in examples for identity in ex["ids"]]
    manifest = {"schema_version": 1, "protocol": "dspigcn-provisional-v1",
                "held_out": held_out, "split": split, "config": asdict(cfg),
                "files_sha256": files, "segments": segments, "windows": len(examples),
                "trajectories": len(identities), "sample_ids": identities,
                "sample_ids_sha256": digest(identities)}
    manifest["manifest_sha256"] = digest(manifest)
    return TrajectoryDataset(examples, manifest)


def collate(examples):
    count = max(len(ex["ids"]) for ex in examples)
    obs = np.zeros((len(examples), len(examples[0]["observations"]), count, 2), np.float32)
    future = np.zeros((len(examples), len(examples[0]["future"]), count, 2), np.float32)
    mask = np.zeros((len(examples), count), bool)
    for i, ex in enumerate(examples):
        n = len(ex["ids"])
        obs[i, :, :n], future[i, :, :n], mask[i, :n] = ex["observations"], ex["future"], True
    return {"observations": torch.from_numpy(obs), "future": torch.from_numpy(future),
            "mask": torch.from_numpy(mask), "examples": examples}
