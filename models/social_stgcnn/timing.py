"""Build clean runtime inputs directly from in-memory neutral trajectories."""

from collections.abc import Iterable

import numpy as np
import torch

from li_junhao_trajectronpp.independent_tpp.data import SceneStore, WindowRecord
from li_junhao_trajectronpp.independent_tpp.protocol import OBSERVATION_POINTS


def timing_batch(store: SceneStore, records: Iterable[WindowRecord]) -> dict[str, torch.Tensor]:
    """Discover complete histories inside this batch and put each target first."""
    groups = {}
    graphs = []
    for row in records:
        key = (row.scene_key, row.anchor)
        if key not in groups:
            frames = range(row.anchor - OBSERVATION_POINTS + 1, row.anchor + 1)
            histories = {}
            for person, track in sorted(store.tracks[row.scene_key].items()):
                if all(frame in track for frame in frames):
                    histories[person] = np.asarray([track[frame][:2] for frame in frames], dtype=np.float32)
            groups[key] = histories

        histories = groups[key]
        target = row.target_id
        ordered = [histories[target]]
        ordered.extend(history for person, history in histories.items() if person != target)
        graphs.append(np.stack(ordered, axis=1))

    if not graphs:
        raise ValueError("A timing batch needs at least one manifest target")
    max_agents = max(graph.shape[1] for graph in graphs)
    positions = np.zeros((len(graphs), OBSERVATION_POINTS, max_agents, 2), dtype=np.float32)
    agent_mask = np.zeros((len(graphs), max_agents), dtype=bool)
    for i, graph in enumerate(graphs):
        agents = graph.shape[1]
        positions[i, :, :agents] = graph
        agent_mask[i, :agents] = True
    return {"positions": torch.from_numpy(positions), "agent_mask": torch.from_numpy(agent_mask)}
