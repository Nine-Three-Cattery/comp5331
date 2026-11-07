import math
import unittest

import torch

from models.social_stgcnn.model import (
    SocialSTGCNN,
    bivariate_nll,
    future_displacements,
    normalized_adjacency,
    sample_world,
)


def observed_tracks():
    t = torch.arange(8, dtype=torch.float32)
    target = torch.stack((2 + t / 4, 1 + t / 8), dim=-1)
    oncoming = torch.stack((5 - t / 4, 2 + t / 16), dim=-1)
    crossing = torch.stack((1 + t / 8, 4 - t / 4), dim=-1)
    return torch.stack((target, oncoming, crossing), dim=1).unsqueeze(0)


class GraphForecastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5331)
        self.model = SocialSTGCNN()

    def test_per_frame_graph_degree_coincidence_and_padding(self):
        positions = torch.tensor([[
            [[0., 0.], [2., 0.], [2., 0.], [float("nan"), 0.]],
            [[0., 0.], [4., 0.], [4., 0.], [0., 0.]],
        ]])
        mask = torch.tensor([[True, True, True, False]])
        actual = normalized_adjacency(positions, mask)
        for t, distance in enumerate((2, 4)):
            target_degree, neighbor_degree = 1 + 2 / distance, 1 + 1 / distance
            edge = 1 / (distance * math.sqrt(target_degree * neighbor_degree))
            expected = torch.tensor([
                [1 / target_degree, edge, edge, 0],
                [edge, 1 / neighbor_degree, 0, 0],
                [edge, 0, 1 / neighbor_degree, 0],
                [0, 0, 0, 0],
            ])
            torch.testing.assert_close(actual[0, t], expected)

    def test_masked_slots_order_and_world_translation(self):
        observed = observed_tracks()
        mask = torch.ones(1, 3, dtype=torch.bool)
        raw = self.model(observed, mask)
        padded = torch.cat((observed, torch.full((1, 8, 4, 2), float("nan"))), dim=2)
        padded_mask = torch.tensor([[True, True, True, False, False, False, False]])
        torch.testing.assert_close(self.model(padded, padded_mask), raw, atol=0, rtol=0)

        order = [0, 2, 1]
        torch.testing.assert_close(self.model(observed[:, :, order], mask[:, order]), raw)
        offset = torch.tensor([13., -9.])
        shifted = self.model(observed + offset, mask)
        torch.testing.assert_close(shifted, raw, atol=0, rtol=0)
        noise = torch.randn(1, 20, 12, 2, generator=torch.Generator().manual_seed(27))
        paths = sample_world(raw, observed[:, -1, 0], noise)
        shifted_paths = sample_world(shifted, observed[:, -1, 0] + offset, noise)
        torch.testing.assert_close(shifted_paths, paths + offset)
        self.assertEqual(paths.shape, (1, 20, 12, 2))
        self.assertTrue(paths.isfinite().all())

    def test_visible_neighbor_motion_reaches_the_target(self):
        observed = observed_tracks()
        changed = observed.clone()
        # Reverse a neighbor's x motion while keeping its last position,
        # the complete target history, and all other tracks the same.
        changed[:, :, 1, 0] = 2 * observed[:, -1:, 1, 0] - observed[:, :, 1, 0]
        mask = torch.ones(1, 3, dtype=torch.bool)
        original, changed_raw = self.model(observed, mask), self.model(changed, mask)
        self.assertGreater((original - changed_raw).abs().max().item(), 1e-5)
        mask[:, 1] = False
        torch.testing.assert_close(self.model(observed, mask), self.model(changed, mask), atol=0, rtol=0)

    def test_cached_full_graph_matches_training_and_uses_distant_peer(self):
        target = observed_tracks()[:, :, :1]
        tracks = torch.cat([target + torch.tensor([float(j), .3*j]) for j in range(12)], dim=2)
        mask = torch.ones(1, 12, dtype=torch.bool)
        cache = self.model.graph_messages(tracks, mask).detach()
        direct = self.model(tracks, mask)
        cached = self.model.from_messages(cache)
        torch.testing.assert_close(direct, cached, rtol=0, atol=0)
        params = tuple(self.model.parameters())
        direct_grad = torch.autograd.grad(direct.square().sum(), params)
        cached_grad = torch.autograd.grad(cached.square().sum(), params)
        for a, b in zip(direct_grad, cached_grad):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # Moving the twelfth node must reach the target; there is no top-k truncation.
        moved = tracks.clone()
        moved[:, :7, 11, 1] += torch.linspace(4., 1., 7)
        self.assertGreater((self.model(moved, mask)-direct).abs().max().item(), 1e-5)

    def test_batched_loss_gradients_include_real_neighbors_and_exclude_padding(self):
        observed = observed_tracks().repeat(2, 1, 1, 1).requires_grad_()
        mask = torch.tensor([[True, True, True], [True, False, False]])
        origin = observed[:, -1, 0].detach()
        step = torch.tensor([[.25, .125], [-.5, .25]])
        times = torch.arange(1, 13)[:, None]
        future = origin[:, None] + times * step[:, None]
        increments = future_displacements(future, origin)
        torch.testing.assert_close(increments, step[:, None].expand(-1, 12, -1))
        raw = self.model(observed, mask)
        self.assertEqual(raw.shape, (2, 12, 5))
        loss = bivariate_nll(raw, increments)
        loss.backward()
        self.assertTrue(loss.isfinite())
        for parameter in self.model.parameters():
            self.assertTrue(parameter.grad.isfinite().all())
        self.assertTrue(observed.grad.isfinite().all())
        self.assertGreater(observed.grad[0, :, 1:].abs().sum().item(), 0)
        self.assertEqual(observed.grad[1, :, 1:].count_nonzero().item(), 0)


class DistributionTests(unittest.TestCase):
    def test_analytic_bivariate_density(self):
        raw = torch.zeros(2, 12, 5, dtype=torch.float64)
        unit_loss = bivariate_nll(raw, torch.zeros(2, 12, 2, dtype=torch.float64))
        self.assertAlmostEqual(unit_loss.item(), math.log(2 * math.pi), places=12)

        # z=(1,-2) with known marginal scales and correlation.
        rho = .6
        raw[..., :2] = torch.tensor([.25, -.5], dtype=raw.dtype)
        raw[..., 2] = math.log(2)
        raw[..., 3] = math.log(.5)
        raw[..., 4] = math.atanh(rho / (1 - 1e-4))
        target = torch.tensor([2.25, -1.5], dtype=raw.dtype).expand(2, 12, 2)
        expected = math.log(2 * math.pi * 2 * .5 * math.sqrt(1 - rho**2))
        expected += (1**2 + (-2)**2 - 2 * rho * 1 * (-2)) / (2 * (1 - rho**2))
        self.assertAlmostEqual(bivariate_nll(raw, target).item(), expected, places=12)

    def test_correlated_draws_and_integration_from_each_origin(self):
        raw = torch.zeros(2, 12, 5, dtype=torch.float64)
        raw[..., 0], raw[..., 1] = .25, -.5
        raw[..., 2], raw[..., 3] = math.log(2), math.log(.5)
        raw[..., 4] = math.atanh(.6 / (1 - 1e-4))
        noise = torch.zeros((2, 20, 12, 2), dtype=raw.dtype)
        noise[:, 1, :, 0] = 1
        noise[:, 2, :, 1] = 1
        origin = torch.tensor([[1., 4.], [-2., .5]], dtype=raw.dtype)
        paths = sample_world(raw, origin, noise)
        times = torch.arange(1, 13, dtype=raw.dtype)[:, None]
        for k, step in enumerate(((.25, -.5), (2.25, -.2), (.25, -.1))):
            expected = origin[:, None] + times * torch.tensor(step, dtype=raw.dtype)
            torch.testing.assert_close(paths[:, k], expected)
        torch.testing.assert_close(future_displacements(paths[:, 0], origin), raw[..., :2])

    def test_extreme_scale_and_correlation_logits_are_finite(self):
        raw = torch.zeros(2, 12, 5)
        raw[:, :, 2], raw[:, :, 3] = -1000, 1000
        raw[0, :, 4], raw[1, :, 4] = -1000, 1000
        raw.requires_grad_()
        loss = bivariate_nll(raw, torch.full((2, 12, 2), .25))
        loss.backward()
        self.assertTrue(loss.isfinite())
        self.assertTrue(raw.grad.isfinite().all())
        paths = sample_world(raw.detach(), torch.zeros(2, 2), torch.ones(2, 20, 12, 2))
        self.assertTrue(paths.isfinite().all())


if __name__ == "__main__":
    unittest.main()
