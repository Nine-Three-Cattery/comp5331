"""The timed window builder must recover the same clean model inputs from neutral rows."""

import unittest

import torch

from li_junhao_trajectronpp.independent_tpp.protocol import CONDITIONS
from models.social_stgcnn.own_unit_tests.test_windows import store, track
from models.social_stgcnn.timing import timing_batch
from models.social_stgcnn.windows import GraphWindows


class TimingBatchTest(unittest.TestCase):
    def test_dense_and_sparse_windows_are_rebuilt_from_visible_positions(self):
        tracks = {str(i): track(i, 0., end=7, dy=.25) for i in range(1, 15)}
        tracks["1"] = track(1., 0., end=32, dy=.25)
        tracks["2"] = track(2., 0., end=19, dy=.25)
        del tracks["13"][3]
        del tracks["14"][7]
        scene = store(tracks, targets=(("2", 7), ("1", 20), ("1", 7)))
        actual = timing_batch(scene, scene.manifests)
        expected = GraphWindows(scene).batch([0, 1, 2], include_future=False)

        self.assertEqual(set(actual), {"positions", "agent_mask", "graph_index", "target_index", "observed_mask"})
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(actual["agent_mask"].sum(dim=1).tolist(), [12, 1])
        self.assertEqual(actual["graph_index"].tolist(), [0, 1, 0])
        self.assertEqual(actual["target_index"].tolist(), [1, 0, 0])
        self.assertEqual(actual["positions"][0, -1, :, 0].tolist(),
                         list(range(1, 13)))

        # Reorder the requests so corruption must use the original manifest indices.
        indices = [2, 0, 1]
        requested = [scene.manifests[i] for i in indices]
        dataset = GraphWindows(scene)
        for _, kind, level in CONDITIONS:
            measured = timing_batch(scene, requested, kind, level, seed=2)
            expected = dataset.batch(indices, kind, level, seed=2, include_future=False)
            for key in measured:
                torch.testing.assert_close(measured[key], expected[key], rtol=0, atol=0)

        # Both dense targets must remain usable when all post-anchor data disappear.
        for trajectory in tracks.values():
            for frame in list(trajectory):
                if frame > 7:
                    del trajectory[frame]
        dense_rows = [scene.manifests[0], scene.manifests[2]]
        truncated = timing_batch(scene, dense_rows)
        torch.testing.assert_close(truncated["positions"][0], actual["positions"][0], rtol=0, atol=0)
        self.assertEqual(truncated["graph_index"].tolist(), [0, 0])
        self.assertEqual(truncated["target_index"].tolist(), [1, 0])

        tracks["10"][4][0] += 5
        refreshed = timing_batch(scene, dense_rows)
        self.assertEqual(refreshed["positions"][0, 4, 9, 0].item(), 15.)
