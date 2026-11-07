"""Construct target-first graphs from the shared metric window manifest."""

from pathlib import Path

import numpy as np
import torch

from li_junhao_trajectronpp.independent_tpp.data import SceneStore, load_split
from li_junhao_trajectronpp.independent_tpp.protocol import (
    OBSERVATION_POINTS, PREDICTION_HORIZON, corruption_seed, perturb_history,
)


HISTORY_OFFSETS = np.arange(1 - OBSERVATION_POINTS, 1)


class GraphWindows:
    """Index neutral positions once, retaining one row per ordered manifest target.

    Each scene/anchor has a shared pool of agents with a complete visible
    history. The graph contains the target and every other member of the pool
    in lexical track ID order. Distance weights are recomputed by the model.
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
        group_ids = {key: i for i, key in enumerate(dict.fromkeys(keys))}
        self._group = np.asarray([group_ids[key] for key in keys], dtype=np.int32)
        members = [{} for _ in group_ids]

        # Every real endpoint refers to its eight positions in this flat bank.
        # The first eight zeros form a complete dummy track for batch padding.
        self._padding_end = OBSERVATION_POINTS - 1
        chunks = [np.zeros((OBSERVATION_POINTS, 2), dtype=np.float64)]
        offset = OBSERVATION_POINTS
        for scene_key, tracks in store.tracks.items():
            for track_id in sorted(tracks):
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
        sizes = np.asarray([len(group) for group in members], dtype=np.int32)
        self.available_neighbors = sizes[self._group] - 1
        self._pool = np.full((len(members), sizes.max(initial=0)), self._padding_end, dtype=np.int32)
        for group, endpoints in enumerate(members):
            self._pool[group, :len(endpoints)] = list(endpoints.values())
        self._target_end = np.asarray([members[g][row.target_id] for g, row in zip(self._group, self.records)],
                                      dtype=np.int32)
        # Preserve original precision until the shared corruption has run.
        self.observed = self._metric[self._target_end[:, None] + HISTORY_OFFSETS]

        # Only supervised labels read past the anchor. A neighbor needs no label.
        self.future = np.empty((len(self.records), PREDICTION_HORIZON, 2), dtype=np.float32)
        for i, row in enumerate(self.records):
            track = store.tracks[row.scene_key][row.target_id]
            horizon = range(row.anchor + 1, row.anchor + 1 + PREDICTION_HORIZON)
            self.future[i] = [track[t][:2] for t in horizon]

    def __len__(self):
        return len(self.records)

    def batch(self, indices, condition="clean", level=0.0, seed=0, *, include_future=True):
        """Build CPU tensors for requested indices. Positions and labels are in meters."""
        indices = np.asarray(indices, dtype=np.intp)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("A batch needs a nonempty sequence of manifest indices")
        histories, observed_masks = zip(*(
            perturb_history(self.observed[i], condition, level, corruption_seed(seed, int(i)))
            for i in indices
        ))
        observed = np.stack(histories)
        groups = self._group[indices]
        neighbors = int(self.available_neighbors[indices].max())
        positions = np.zeros((len(indices), OBSERVATION_POINTS, 1 + neighbors, 2), dtype=np.float32)
        agent_mask = np.ones((len(indices), 1 + neighbors), dtype=bool)
        positions[:, :, 0] = observed
        if neighbors:
            pool = self._pool[groups, :neighbors + 1]
            # Skip each target's original slot in the lexical pool. Remaining
            # agents keep their order, and existing dummy endpoints stay last.
            target_slot = (pool == self._target_end[indices, None]).argmax(axis=1)
            slots = np.arange(neighbors)[None, :]
            ends = np.take_along_axis(pool, slots + (slots >= target_slot[:, None]), axis=1)
            positions[:, :, 1:] = self._metric[ends[:, :, None] + HISTORY_OFFSETS].transpose(0, 2, 1, 3)
            agent_mask[:, 1:] = ends != self._padding_end
        batch = {
            "positions": torch.from_numpy(positions),
            "agent_mask": torch.from_numpy(agent_mask),
            "target_index": torch.zeros(len(indices), dtype=torch.long),
            "observed_mask": torch.from_numpy(np.stack(observed_masks)),
        }
        if include_future:
            batch["future"] = torch.from_numpy(self.future[indices])
        return batch
