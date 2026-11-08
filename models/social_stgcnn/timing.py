"""Discover full scene histories on demand from in-memory neutral coordinates."""

from collections.abc import Iterable

import numpy as np
import torch

from li_junhao_trajectronpp.independent_tpp.data import SceneStore, WindowRecord
from li_junhao_trajectronpp.independent_tpp.protocol import OBSERVATION_POINTS
from .windows import graph_layout, observation_batch


def timing_batch(store: SceneStore, records: Iterable[WindowRecord], condition="clean", level=0.0,
                 seed=0, *, manifest_indices=None) -> dict[str, torch.Tensor]:
    """Discover each scene's visible agents during timing and apply the target corruption."""
    records = list(records)
    if not records:
        raise ValueError("A timing batch needs at least one manifest target")
    keys = [(r.scene_key, r.anchor) for r in records]
    scene_keys = sorted(set(keys))
    group_id = {key: i for i, key in enumerate(scene_keys)}
    scenes, slots = [], []
    for scene_key, anchor in scene_keys:
        frames = range(anchor + 1 - OBSERVATION_POINTS, anchor + 1)
        tracks = store.tracks[scene_key]
        visible = [person for person in sorted(tracks, key=int) if all(t in tracks[person] for t in frames)]
        slots.append({person: i for i, person in enumerate(visible)})
        scenes.append(np.asarray([[tracks[person][t][:2] for person in visible] for t in frames], dtype=np.float64))

    requested = np.asarray([group_id[key] for key in keys], dtype=np.intp)
    target_slots = [slots[g][row.target_id] for g, row in zip(requested, records)]
    groups, graph_index = graph_layout(requested, condition)
    widths = np.asarray([scenes[g].shape[1] for g in groups], dtype=np.intp)
    positions = np.zeros((len(groups), OBSERVATION_POINTS, widths.max(), 2), dtype=np.float64)
    for j, g in enumerate(groups):
        positions[j, :, :widths[j]] = scenes[g]
    if manifest_indices is None:
        if condition == "clean":
            manifest_indices = range(len(records))  # The clean transformation draws no random numbers.
        else:
            lookup = {r.window_id: i for i, r in enumerate(store.manifests)}
            manifest_indices = [lookup[r.window_id] for r in records]
    return observation_batch(positions, widths, graph_index, target_slots, manifest_indices, condition, level, seed)
