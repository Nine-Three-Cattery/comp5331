"""Check the units used for optimization and validation against hand calculations."""

import unittest
from unittest.mock import patch

import torch

from models.social_stgcnn.run import train_epoch, validation_nll


class SceneWeightTests(unittest.TestCase):
    def test_each_scene_contributes_including_the_update_boundary_and_tail(self):
        weight = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            weight.weight.zero_()
        optimizer = torch.optim.SGD(weight.parameters(), lr=0.1)
        # First group: mean of gradients -2 and -6 gives weight 0.4.
        # Last group: one gradient 2*(0.4-5) takes weight to 1.32.
        def quadratic(model, target):
            return (model.weight[0, 0] - target).square()

        with patch("models.social_stgcnn.run.scene_loss", side_effect=quadratic):
            loss = train_epoch(weight, optimizer, [1., 3., 5.], [0, 1, 2], 2)
        self.assertAlmostEqual(weight.weight.item(), 1.32)
        self.assertAlmostEqual(loss, (1 + 9 + 4.6 ** 2) / 3)

    def test_validation_weights_targets_equally_across_scene_sizes(self):
        model = torch.nn.Identity()
        scenes = [(0, None, [0], None), (4, None, [0, 1, 2], None)]
        with patch("models.social_stgcnn.run.scene_loss", side_effect=lambda _, scene: torch.tensor(scene[0])):
            self.assertEqual(validation_nll(model, scenes), 3.0)
