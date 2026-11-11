"""Owner graph adapter on the group's neutral, hash-locked loader."""
from functools import lru_cache

import numpy as np
import torch

from independent_tpp.protocol import corruption_seed, perturb_history


class GraphDataset:
    def __init__(self, store, seed=0, condition='clean', level=0, radius=3.0):
        self.store, self.seed = store, seed
        self.condition, self.level, self.radius = condition, level, radius

    def __len__(self):
        return len(self.store.manifests)

    @lru_cache(maxsize=4096)
    def context(self, scene_key, anchor):
        # Eligibility is history-only, including pedestrians without ANY future.
        history = range(anchor-7, anchor+1)
        ids, positions = [], []
        for identifier, track in sorted(self.store.tracks[scene_key].items()):
            if all(frame in track for frame in history):
                ids.append(identifier)
                positions.append(np.stack([track[frame][:2] for frame in history]))
        return ids, np.stack(positions)

    def inputs(self, index):
        row = self.store.manifests[index]
        ids, context = self.context(row.scene_key, row.anchor)
        target = ids.index(row.target_id)
        observed, observed_mask = perturb_history(context[target], self.condition, self.level,
                                                  corruption_seed(self.seed, index))
        distances = np.linalg.norm(context-observed[None], axis=-1)
        selected = np.flatnonzero((distances <= self.radius).any(axis=1))
        selected = selected[selected != target]
        positions = np.concatenate((observed[None], context[selected]), axis=0).transpose(1, 0, 2)
        origin = observed[-1].copy()
        return dict(positions=(positions-origin).astype('float32'),
                    origin=origin.astype('float32'), observed=observed.astype('float32'),
                    observed_mask=observed_mask, index=index)

    def __getitem__(self, index):
        item = self.inputs(index)
        row = self.store.manifests[index]
        track = self.store.tracks[row.scene_key][row.target_id]
        # Label assembly is separate from input construction and never enters it.
        future = np.stack([track[t][:2] for t in range(row.anchor+1, row.anchor+13)])
        return {**item, 'future': future.astype('float32')}


def collate(items):
    b, n = len(items), max(x['positions'].shape[1] for x in items)
    positions = np.zeros((b, 8, n, 2), np.float32)
    mask = np.zeros((b, n), bool)
    for i, item in enumerate(items):
        count = item['positions'].shape[1]
        positions[i, :, :count] = item['positions']
        mask[i, :count] = True
    batch = {key: torch.from_numpy(np.stack([x[key] for x in items]))
             for key in ('origin', 'observed', 'observed_mask')}
    if 'future' in items[0]:
        batch['future'] = torch.from_numpy(np.stack([x['future'] for x in items]))
    return {**batch, 'positions': torch.from_numpy(positions), 'mask': torch.from_numpy(mask),
            'indices': [x['index'] for x in items]}


def batches(items, batch_size, indices=None):
    indices = list(range(len(items))) if indices is None else list(indices)
    for start in range(0, len(indices), batch_size):
        yield collate([items[int(i)] for i in indices[start:start+batch_size]])
