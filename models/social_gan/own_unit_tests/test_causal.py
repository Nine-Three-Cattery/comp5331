"""Regression checks for causal social inputs and the forecast contract."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

MODEL = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(MODEL), str(MODEL.parents[1] / "li_junhao_trajectronpp")]

from data import HistoryIndex, scene_latents
from independent_tpp.data import SceneStore, WindowRecord
from model import Discriminator, Generator, displacements, to_world, train_batch


def store(include_future, offset=0):
    track_a = {}
    track_b = {}
    newcomer = {}
    for t in range(20 if include_future else 8):
        poison = 9000 if t >= 8 else 0
        track_a[t] = np.array([offset + t * 0.4 + poison, offset + 2, 900, 900, 900, 900])
        track_b[t] = np.array([offset + t * 0.4 + poison, offset + 4, -900, -900, -900, -900])
        if t >= 8:
            newcomer[t] = track_b[t].copy()
    scene = ("eth", "source")
    return SceneStore(
        {scene: {"1": track_a, "2": track_b, "3": newcomer}},
        [WindowRecord({"target_id": "1", "scene_t": 7, "scene_id": scene[0], "source_file": scene[1]}),
         WindowRecord({"target_id": "2", "scene_t": 7, "scene_id": scene[0], "source_file": scene[1]})],
        {"fold": "eth", "split": "test"},
    )


class CausalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = Generator(8)
        self.indices = np.array([0, 1])

    def test_target_and_neighbor_futures_are_invisible(self):
        full = store(True)
        past = store(False)
        for track in past.tracks["eth", "source"].values():
            for state in track.values():
                state[2:] = -state[2:] * 100
        a, b = (HistoryIndex(x, 6, 12) for x in (full, past))
        for kind, level in (("clean", 0), ("noise", 0.2), ("missing", 2)):
            inputs, origin, observed, mask = a.pack(self.indices, 0, kind, level)
            other, other_origin, other_obs, other_mask = b.pack(self.indices, 0, kind, level)
            for left, right in zip(inputs, other):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            for left, right in ((origin, other_origin), (observed, other_obs), (mask, other_mask)):
                np.testing.assert_array_equal(left, right)
            latent, seeds = scene_latents(a, 0, 20, 8)
            self.assertEqual(seeds[0], seeds[1])
            torch.testing.assert_close(latent[0], latent[1], rtol=0, atol=0)
            torch.testing.assert_close(self.model(*inputs, latent), self.model(*other, latent), rtol=0, atol=0)
        self.assertTrue(bool(inputs[3][:, 0].all()))
        self.assertEqual(int(inputs[3].sum()), 2)
        self.assertFalse(bool(mask[:, -2:].any()))

    def test_visible_neighbor_history_changes_the_forecast(self):
        original = store(True)
        changed = store(True)
        neighbor = changed.tracks["eth", "source"]["2"]
        for frame in range(7):
            neighbor[frame][0] -= (7 - frame) * 0.25
        original_index, changed_index = (HistoryIndex(item, 6, 12) for item in (original, changed))
        target_index = np.array([0])
        before, origin, observed, mask = original_index.pack(target_index)
        after, changed_origin, changed_observed, changed_mask = changed_index.pack(target_index)
        for left, right in ((origin, changed_origin), (observed, changed_observed), (mask, changed_mask)):
            np.testing.assert_array_equal(left, right)
        for slot in (0, 2, 3):
            torch.testing.assert_close(before[slot], after[slot], rtol=0, atol=0)
        self.assertTrue(bool(before[3][0, 0]))
        self.assertFalse(torch.equal(before[1][0, 0], after[1][0, 0]))
        latent, _ = scene_latents(original_index, 0, 20, 8)
        self.assertFalse(torch.equal(self.model(*before, latent[:1]), self.model(*after, latent[:1])))

    def test_coordinate_round_trip_and_displacements(self):
        a, b = (HistoryIndex(store(False, shift), 6, 12) for shift in (0, 100))
        inputs, origin, _, _ = a.pack(self.indices)
        other, other_origin, _, _ = b.pack(self.indices)
        steps = displacements(inputs[0])
        torch.testing.assert_close(steps[:, 1:, 0], torch.full((2, 7), 0.4))
        torch.testing.assert_close(steps[:, 0], torch.zeros(2, 2))
        torch.testing.assert_close(to_world(steps[:, None], inputs[0][:, 0])[:, 0], inputs[0])
        latent, _ = scene_latents(a, 0, 20, 8)
        samples = self.model(*inputs, latent) + torch.from_numpy(origin)[:, None, None]
        shifted = self.model(*other, latent) + torch.from_numpy(other_origin)[:, None, None]
        torch.testing.assert_close(shifted - samples, torch.full_like(shifted, 100))
        self.assertEqual(samples.shape, (2, 20, 12, 2))
        self.assertTrue(bool(torch.isfinite(samples).all()))
        self.assertTrue(bool((samples[:, 0] != samples[:, 1]).any()))

    def test_isolated_target_and_adversarial_update(self):
        index = HistoryIndex(store(True), 0.01, 12)
        inputs, origin, _, _ = index.pack(self.indices)
        self.assertFalse(bool(inputs[3].any()))
        latent, _ = scene_latents(index, 0, 20, 8)
        discriminator = Discriminator()
        before = self.model.output.weight.detach().clone()
        losses = train_batch(
            self.model, discriminator, torch.optim.Adam(self.model.parameters()),
            torch.optim.Adam(discriminator.parameters()), inputs,
            torch.from_numpy(index.future() - origin[:, None]), latent, 0.05,
        )
        self.assertTrue(bool(torch.isfinite(losses).all()))
        self.assertFalse(torch.equal(before, self.model.output.weight))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
