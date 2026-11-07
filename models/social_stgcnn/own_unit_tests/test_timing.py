"""The timed window builder must recover the same clean model inputs from neutral rows."""

import unittest

import torch

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

        self.assertEqual(set(actual), {"positions", "agent_mask"})
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(actual["agent_mask"].sum(dim=1).tolist(), [12, 1, 12])
        self.assertEqual(actual["positions"][0, -1, :, 0].tolist(),
                         [2, 1, 10, 11, 12, 3, 4, 5, 6, 7, 8, 9])

        # Both dense targets must remain usable when all post-anchor data disappear.
        for trajectory in tracks.values():
            for frame in list(trajectory):
                if frame > 7:
                    del trajectory[frame]
        dense_rows = [scene.manifests[0], scene.manifests[2]]
        truncated = timing_batch(scene, dense_rows)
        for key in truncated:
            torch.testing.assert_close(truncated[key], actual[key][[0, 2]], rtol=0, atol=0)

        tracks["10"][4][0] += 5
        refreshed = timing_batch(scene, dense_rows)
        self.assertEqual(refreshed["positions"][0, 4, 2, 0].item(), 15.)
