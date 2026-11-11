"""Construct causal model inputs from the shared neutral positions."""

from collections import defaultdict
import hashlib

import numpy as np
import torch

from independent_tpp.protocol import corruption_seed, perturb_history


class HistoryIndex:
    def __init__(self, store, radius, neighbor_limit):
        self.store = store
        self.radius = radius
        self.neighbor_limit = neighbor_limit
        self.by_anchor = defaultdict(list)
        for scene, tracks in store.tracks.items():
            anchors = {row.anchor for row in store.manifests if row.scene_key == scene}
            for pedestrian, track in tracks.items():
                for anchor in anchors.intersection(track):
                    frames = range(anchor - 7, anchor + 1)
                    if all(frame in track for frame in frames):
                        history = np.array([track[frame][:2] for frame in frames])
                        self.by_anchor[scene, anchor].append((pedestrian, history))
        self.observed = np.array([
            [store.state(row.scene_key, row.target_id, t)[:2] for t in range(row.anchor - 7, row.anchor + 1)]
            for row in store.manifests
        ])
        self.groups = [(*row.scene_key, row.anchor) for row in store.manifests]

    def future(self):
        return np.array([
            [self.store.state(row.scene_key, row.target_id, t)[:2] for t in range(row.anchor + 1, row.anchor + 13)]
            for row in self.store.manifests
        ], dtype=np.float32)

    def pack(self, indices, seed=0, condition="clean", level=0):
        count = len(indices)
        shape = (count, self.neighbor_limit)
        neighbors = np.zeros((*shape, 8, 2), dtype=np.float32)
        relative = np.zeros((*shape, 2), dtype=np.float32)
        present = np.zeros(shape, dtype=bool)
        target = self.observed[indices].copy()
        masks = np.ones((count, 8), dtype=bool)
        for batch_index, index in enumerate(indices):
            row = self.store.manifests[index]
            if condition != "clean":
                target[batch_index], masks[batch_index] = perturb_history(
                    target[batch_index], condition, level, corruption_seed(seed, int(index))
                )
            origin = target[batch_index, -1]
            nearby = []
            for pedestrian, history in self.by_anchor[row.scene_key, row.anchor]:
                if pedestrian != row.target_id:
                    offset = history[-1] - origin
                    distance_sq = float(offset @ offset)
                    if distance_sq <= self.radius ** 2:
                        nearby.append((distance_sq, pedestrian, history, offset))
            nearby.sort(key=lambda neighbor: neighbor[:2])
            for slot, (_, _, history, offset) in enumerate(nearby[:self.neighbor_limit]):
                # Center neighbor history on its own endpoint. The separate
                # relative endpoint retains the geometry used by the pool.
                neighbors[batch_index, slot] = history - history[-1]
                relative[batch_index, slot] = offset
                present[batch_index, slot] = True
        origin = target[:, -1].astype(np.float32)
        centered = (target - target[:, -1:]).astype(np.float32)
        tensors = tuple(torch.from_numpy(array) for array in (centered, neighbors, relative, present))
        return tensors, origin, target.astype(np.float32), masks


def draw_scene_latent(fold, split, seed, key, count, dimension):
    identity = repr((fold, split, seed, key)).encode()
    sampling_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
    values = np.random.default_rng(sampling_seed).standard_normal((count, dimension), dtype=np.float32)
    return values, sampling_seed


def scene_latents(index, seed, count, dimension):
    """Couple equal sample indices within the same scene and anchor."""
    fold = index.store.identity["fold"]
    split = index.store.identity["split"]
    by_group = {
        key: draw_scene_latent(fold, split, seed, key, count, dimension)
        for key in dict.fromkeys(index.groups)
    }
    return (
        torch.from_numpy(np.array([by_group[key][0] for key in index.groups])),
        [by_group[key][1] for key in index.groups],
    )
