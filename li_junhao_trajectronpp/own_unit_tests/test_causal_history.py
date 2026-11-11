"""Regression for target and neighbor future leakage through derivatives."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from independent_tpp.data import SceneStore, WindowRecord, build_window, collate_windows
from independent_tpp.model import IndependentTrajectronPP
from independent_tpp.protocol import CONDITIONS, DT, history_state, perturb_history


def test_derivatives_and_fill():
    t = np.arange(8) * DT
    positions = np.column_stack((t ** 2, 2 * t))
    states = history_state(positions)
    np.testing.assert_allclose(states[0, 2:], 0)
    np.testing.assert_allclose(states[1, 4:], 0)
    np.testing.assert_allclose(states[2:, 4:], np.tile([2, 0], (6, 1)), atol=1e-12)
    filled, mask = perturb_history(positions, "missing", 2, 0)
    np.testing.assert_array_equal(mask, [True] * 6 + [False] * 2)
    np.testing.assert_allclose(filled[-1], positions[5] + 2 * (positions[5] - positions[4]))
    np.testing.assert_array_equal(
        perturb_history(positions, "noise", 0.2, 42)[0],
        perturb_history(positions, "noise", 0.2, 42)[0],
    )


def test_future_invariance():
    scene = ("eth", "fixture")
    record = WindowRecord({"scene_id": "eth", "source_file": "fixture", "target_id": "1",
                           "scene_t": 7, "anchor_t": 7, "window_id": "fixture:1:7"})

    def store(changed):
        tracks = {}
        for agent in ("1", "2"):
            positions = np.column_stack((np.arange(20) * DT, np.full(20, int(agent))))
            if changed == agent:
                positions[8:, 0] += 1
            tracks[agent] = dict(enumerate(history_state(positions)))
        return SceneStore({scene: tracks}, [record])

    torch.manual_seed(7)
    model = IndependentTrajectronPP().eval()

    def predict(window):
        batch = collate_windows([window])
        features = {key: batch[key] for key in ("target_history", "neighbors", "neighbor_mask")}
        torch.manual_seed(42)
        with torch.no_grad():
            return model.sample_positions(features, num_samples=20) + batch["origin_metric"][:, None, None]

    for _, condition, level in CONDITIONS:
        original = build_window(store(None), record, condition, level, 42)
        expected = predict(original)
        np.testing.assert_allclose(
            original["target_history"][:, :2] + original["origin_metric"],
            original["observed_metric"], atol=1e-6,
        )
        for agent in ("1", "2"):
            changed = build_window(store(agent), record, condition, level, 42)
            for key in ("target_history", "neighbors", "neighbor_mask", "observed_metric", "observed_mask"):
                np.testing.assert_array_equal(original[key], changed[key])
            torch.testing.assert_close(expected, predict(changed), rtol=0, atol=0)
            if agent == "1":
                assert not np.array_equal(original["future_metric"], changed["future_metric"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    test_derivatives_and_fill()
    test_future_invariance()
    print("CAUSAL_HISTORY_OK target and neighbor, six conditions")
