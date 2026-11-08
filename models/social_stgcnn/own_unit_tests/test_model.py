"""Numerical fixtures for the full-scene release operators."""

import math
import unittest

import torch

from models.social_stgcnn.model import (
    SocialSTGCNN, bivariate_nll, future_displacements, release_displacements, sample_world, scene_geometry,
)


class GeometryTests(unittest.TestCase):
    def test_increment_graph_has_self_loops_and_zero_weight_for_equal_motion(self):
        expected_steps = torch.zeros(1, 8, 3, 2, dtype=torch.float64)
        expected_steps[:, 1, 1:, 0] = 2.
        expected_steps[:, 2, 1:, 0] = 4.
        expected_steps[:, 3:, :, 1] = .25
        starts = torch.tensor([[-8., 0.], [1., 9.], [5., -4.]], dtype=torch.float64)
        positions = starts + expected_steps.cumsum(dim=1)
        steps, graph = scene_geometry(positions)
        torch.testing.assert_close(steps, expected_steps.float())
        torch.testing.assert_close(graph[:, 0], torch.zeros_like(graph[:, 0]), rtol=0, atol=0)
        torch.testing.assert_close(graph[:, 3:], torch.zeros_like(graph[:, 3:]), rtol=0, atol=0)
        for t, distance in ((1, 2.), (2, 4.)):
            center, neighbor = 1 + 2 / distance, 1 + 1 / distance
            edge = -1 / (distance * math.sqrt(center * neighbor))
            expected = torch.tensor([
                [1 - 1 / center, edge, edge],
                [edge, 1 - 1 / neighbor, 0.],
                [edge, 0., 1 - 1 / neighbor],
            ])
            torch.testing.assert_close(graph[0, t], expected)
        shifted_steps, shifted_graph = scene_geometry(positions + torch.tensor([30., -11.]))
        torch.testing.assert_close(shifted_steps, steps, rtol=0, atol=0)
        torch.testing.assert_close(shifted_graph, graph, rtol=0, atol=0)

    def test_four_decimal_rounding_precedes_difference_and_float32_conversion(self):
        positions = torch.full((1, 8, 1, 2), 100000.12344, dtype=torch.float64)
        positions[:, 1:, 0, 0] = 100000.12346
        steps, graph = scene_geometry(positions)
        self.assertEqual(steps.dtype, torch.float32)
        self.assertEqual(graph.shape, (1, 8, 1, 1))
        self.assertEqual(graph.count_nonzero().item(), 0)
        self.assertAlmostEqual(steps[0, 1, 0, 0].item(), .0001, places=9)
        torch.testing.assert_close(release_displacements(positions[:, 1:], positions[:, 0]), steps[:, 1:])
        self.assertAlmostEqual(future_displacements(positions[:, 1:], positions[:, 0])[0, 0, 0, 0].item(),
                               .00002, places=9)


class FullSceneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5331)
        self.model = SocialSTGCNN()

    def test_single_person_and_batched_full_scene_shapes(self):
        self.model.eval()
        for agents in (1, 3, 11):
            positions = torch.randn(2, 8, agents, 2, dtype=torch.float64)
            steps, graph = scene_geometry(positions)
            batched = self.model.full_scene(steps, graph)
            self.assertEqual(batched.shape, (2, 12, agents, 5))
            self.assertTrue(batched.isfinite().all())
            for i in range(2):
                alone = self.model(steps[i:i + 1], graph[i:i + 1])
                torch.testing.assert_close(batched[i], alone[0])
        # With one pedestrian the graph branch has a zero Laplacian.
        steps, graph = scene_geometry(torch.randn(1, 8, 1, 2, dtype=torch.float64))
        self.assertGreater((self.model(steps, graph) - self.model(steps * 0, graph)).abs().max().item(), 1e-5)

    def test_decoder_uses_other_agents_even_without_graph_messages(self):
        self.model.eval()
        steps = torch.randn(1, 8, 4, 2)
        graph = torch.zeros(1, 8, 4, 4)
        before = self.model(steps, graph)
        changed = steps.clone()
        changed[0, 1:, 1, 0] += 1
        after = self.model(changed, graph)
        # Agent zero never changed. With L=0 only TXP can carry the effect
        # from agent one's temporal embedding to agent zero's output.
        self.assertGreater((before[:, :, 0] - after[:, :, 0]).abs().max().item(), 1e-5)

    def test_agent_beyond_seven_neighbors_contributes_through_the_full_graph(self):
        self.model.eval()
        positions = torch.randn(1, 8, 12, 2, dtype=torch.float64)
        first = self.model(*scene_geometry(positions))
        positions[:, 1:, -1, 1] += torch.linspace(.5, 4., 7)
        changed = self.model(*scene_geometry(positions))
        self.assertGreater((first[:, :, 0] - changed[:, :, 0]).abs().max().item(), 1e-5)

    def test_storage_order_across_both_time_feature_reshapes(self):
        model = self.model.eval().double()
        # Route positive, labelled scalars through the input bypass and the
        # centre pixels of the first and last convolutions. All residual
        # refinements then reduce to the identity.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.bypass_norm.weight.fill_(math.sqrt(1 + model.bypass_norm.eps))
            for channel in range(5):
                model.bypass_projection.weight[channel, 0, 0, 0] = channel + 1
            for channel in range(12):
                model.expand.weight[channel, channel % 8, 1, 1] = 1
                model.readout.weight[channel, channel, 1, 1] = 1
        steps = torch.arange(1, 49, dtype=torch.float64).reshape(1, 8, 3, 2)
        actual = model(steps, steps.new_zeros(1, 8, 3, 3))
        expected = torch.empty_like(actual)
        for parameter in range(5):
            for time in range(12):
                output_channel, feature = divmod(parameter * 12 + time, 5)
                encoder_channel, observed_time = divmod((output_channel % 8) * 5 + feature, 8)
                expected[:, time, :, parameter] = (encoder_channel + 1) * steps[:, observed_time, :, 0]
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)

    def test_training_reaches_every_registered_parameter(self):
        self.assertEqual(sum(value.numel() for value in self.model.parameters()), 6254)
        steps, graph = scene_geometry(torch.randn(2, 8, 4, 2, dtype=torch.float64))
        raw = self.model(steps, graph)
        loss = bivariate_nll(raw, torch.rand(2, 12, 4, 2))
        loss.backward()
        self.assertTrue(loss.isfinite())
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(parameter.grad.isfinite().all(), name)


class DistributionTests(unittest.TestCase):
    def test_analytic_density_and_released_floor(self):
        raw = torch.zeros(2, 12, 5, dtype=torch.float64)
        zero = torch.zeros(2, 12, 2, dtype=raw.dtype)
        self.assertAlmostEqual(bivariate_nll(raw, zero).item(), math.log(2 * math.pi), places=12)
        rho = .6
        raw[..., 0], raw[..., 1] = .25, -.5
        raw[..., 2], raw[..., 3], raw[..., 4] = math.log(2), math.log(.5), math.atanh(rho)
        target = zero + torch.tensor([2.25, -1.5], dtype=raw.dtype)
        expected = math.log(2 * math.pi * math.sqrt(1 - rho**2)) + (5 + 4 * rho) / (2 * (1 - rho**2))
        self.assertAlmostEqual(bivariate_nll(raw, target).item(), expected, places=12)
        raw.requires_grad_()
        floored = bivariate_nll(raw, target + 1000)
        self.assertAlmostEqual(floored.item(), -math.log(1e-20), places=12)
        floored.backward()
        self.assertEqual(raw.grad.count_nonzero().item(), 0)

    def test_correlated_draws_and_integration_from_original_metric_origin(self):
        raw = torch.zeros(2, 12, 5, dtype=torch.float64)
        raw[..., 0], raw[..., 1] = .25, -.5
        raw[..., 2], raw[..., 3], raw[..., 4] = math.log(2), math.log(.5), math.atanh(.6)
        noise = torch.zeros(2, 20, 12, 2, dtype=raw.dtype)
        noise[:, 1, :, 0] = 1
        noise[:, 2, :, 1] = 1
        origin = torch.tensor([[1.000023, 4.], [-2., .5]], dtype=raw.dtype)
        paths = sample_world(raw, origin, noise)
        self.assertEqual(paths.shape, (2, 20, 12, 2))
        times = torch.arange(1, 13, dtype=raw.dtype)[:, None]
        for k, step in enumerate(((.25, -.5), (2.25, -.2), (.25, -.1))):
            expected = origin[:, None] + times * torch.tensor(step, dtype=raw.dtype)
            torch.testing.assert_close(paths[:, k], expected)
        torch.testing.assert_close(future_displacements(paths[:, 0], origin), raw[..., :2])


if __name__ == "__main__":
    unittest.main()
