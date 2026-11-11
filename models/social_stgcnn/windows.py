"""Unpadded scene windows and manifest-aligned targets for Social-STGCNN."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from li_junhao_trajectronpp.independent_tpp.data import SceneStore, load_split
from li_junhao_trajectronpp.independent_tpp.protocol import (
    OBSERVATION_POINTS, PREDICTION_HORIZON, corruption_seed, perturb_history,
)
from .model import release_displacements


HISTORY_OFFSETS = np.arange(1 - OBSERVATION_POINTS, 1)


@dataclass
class CleanScene:
    scene_key: tuple[str, str]
    anchor: int
    positions: torch.Tensor         # [8, A, 2], float64, numeric pedestrian ID order
    target_slots: torch.Tensor      # [S] supervised slots in this scene
    future_steps: torch.Tensor      # [S, 12, 2] rounded-coordinate increments in meters
    manifest_indices: torch.Tensor  # [S] original ordered target rows


def graph_layout(groups, condition):
    """A clean scene serves all its targets; a perturbed graph serves one."""
    if condition == "clean":
        return np.unique(groups, return_inverse=True)
    return groups, np.arange(len(groups), dtype=np.intp)


def observation_batch(positions, widths, graph_index, target_slots, indices, condition, level, seed):
    """Apply the shared perturbation only to each target's actual slot."""
    histories = positions[graph_index, :, target_slots]
    supplied, masks = zip(*(
        perturb_history(xy, condition, level, corruption_seed(seed, int(i)))
        for xy, i in zip(histories, indices)
    ))
    positions[graph_index, :, target_slots] = np.stack(supplied)
    return {
        "positions": torch.from_numpy(positions),
        "agent_mask": torch.from_numpy(np.arange(positions.shape[2])[None, :] < widths[:, None]),
        "graph_index": torch.from_numpy(np.asarray(graph_index, dtype=np.int64)),
        "target_index": torch.from_numpy(np.asarray(target_slots, dtype=np.int64)),
        "observed_mask": torch.from_numpy(np.stack(masks)),
    }


class GraphWindows:
    """Index neutral positions once, retaining one row per ordered manifest target.

    Each scene/anchor has a shared pool of agents with a complete visible
    history. All nodes are kept in numeric pedestrian ID order. A target has
    the same slot as it would when another pedestrian is being predicted.
    A neighbor can leave the scene immediately after the anchor.
    """

    @classmethod
    def from_split(cls, fold: str, split: str, data_root: Path, max_neighbors=None):
        return cls(load_split(fold, split, str(data_root)), max_neighbors)

    def __init__(self, store: SceneStore, max_neighbors=None):
        if max_neighbors is not None:
            raise ValueError("The full graph requires max_neighbors=None")
        self.records = store.manifests
        self.identity = dict(store.identity)

        keys = [(row.scene_key, row.anchor) for row in self.records]
        self._scene_keys = sorted(set(keys))
        group_ids = {key: i for i, key in enumerate(self._scene_keys)}
        self._group = np.asarray([group_ids[key] for key in keys], dtype=np.int32)
        members = [{} for _ in group_ids]

        # Every real endpoint refers to its eight positions in this flat bank.
        # The first eight zeros form a complete dummy track for batch padding.
        self._padding_end = OBSERVATION_POINTS - 1
        chunks = [np.zeros((OBSERVATION_POINTS, 2), dtype=np.float64)]
        offset = OBSERVATION_POINTS
        for scene_key, tracks in store.tracks.items():
            for track_id in sorted(tracks, key=int):
                track = tracks[track_id]
                frames = np.asarray(sorted(track), dtype=np.int64)
                chunks.append(np.asarray([track[t][:2] for t in frames], dtype=np.float64))
                # Unique sorted frame numbers span exactly seven intervals iff
                # all eight time points are present, even in a gapped track.
                spans = frames[OBSERVATION_POINTS - 1:] - frames[:1 - OBSERVATION_POINTS]
                for end in np.flatnonzero(spans == OBSERVATION_POINTS - 1) + OBSERVATION_POINTS - 1:
                    group = group_ids.get((scene_key, int(frames[end])))
                    if group is not None:
                        members[group][track_id] = offset + int(end)
                offset += len(frames)
        self._metric = np.concatenate(chunks)
        self._widths = np.asarray([len(group) for group in members], dtype=np.int32)
        self.available_neighbors = self._widths[self._group] - 1
        self._pool = np.full((len(members), self._widths.max(initial=0)), self._padding_end, dtype=np.int32)
        slot_by_id = []
        for group, endpoints in enumerate(members):
            self._pool[group, :len(endpoints)] = list(endpoints.values())
            slot_by_id.append({person: slot for slot, person in enumerate(endpoints)})
        self._slots = np.asarray([slot_by_id[g][row.target_id] for g, row in zip(self._group, self.records)],
                                 dtype=np.int64)
        target_end = self._pool[self._group, self._slots]
        # Preserve original precision until the shared corruption has run.
        self.observed = self._metric[target_end[:, None] + HISTORY_OFFSETS]
        ordered_rows = np.lexsort((self._slots, self._group))
        self._scene_rows = np.split(ordered_rows, np.bincount(self._group).cumsum()[:-1])

        # Only supervised labels read past the anchor. A neighbor needs no label.
        metric_futures = np.empty((len(self.records), PREDICTION_HORIZON, 2), dtype=np.float64)
        for i, row in enumerate(self.records):
            track = store.tracks[row.scene_key][row.target_id]
            horizon = range(row.anchor + 1, row.anchor + 1 + PREDICTION_HORIZON)
            metric_futures[i] = [track[t][:2] for t in horizon]
        self.future = metric_futures.astype(np.float32)
        self.future_steps = release_displacements(torch.from_numpy(metric_futures),
                                                  torch.from_numpy(self.observed[:, -1])).numpy()

    def __len__(self):
        return len(self.records)

    def clean_scenes(self):
        """Yield each scene/anchor once, including all eligible target labels."""
        for group, ((key, anchor), rows) in enumerate(zip(self._scene_keys, self._scene_rows)):
            ends = self._pool[group, :self._widths[group]]
            xy = self._metric[ends[:, None] + HISTORY_OFFSETS].transpose(1, 0, 2).copy()
            yield CleanScene(key, anchor, torch.from_numpy(xy), torch.from_numpy(self._slots[rows]),
                             torch.from_numpy(self.future_steps[rows]), torch.from_numpy(rows))

    def batch(self, indices, condition="clean", level=0.0, seed=0, *, include_future=True):
        """Pack metric histories and map each requested target to its graph and node."""
        indices = np.asarray(indices, dtype=np.intp)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("A batch needs a nonempty sequence of manifest indices")
        groups, graph_index = graph_layout(self._group[indices], condition)
        widths = self._widths[groups]
        ends = self._pool[groups, :widths.max()]
        positions = self._metric[ends[:, :, None] + HISTORY_OFFSETS].transpose(0, 2, 1, 3)
        batch = observation_batch(positions, widths, graph_index, self._slots[indices],
                                  indices, condition, level, seed)
        if include_future:
            batch["future"] = torch.from_numpy(self.future[indices])
        return batch
