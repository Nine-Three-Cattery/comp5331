"""Regression checks for social max pooling and adversarial variety training."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Generator, train_batch


class RelativeMessages(nn.Module):
    def forward(self, features):
        return features[:, -2:].repeat(1, 8)


class PoolingTests(unittest.TestCase):
    def test_max_pool_is_order_invariant_and_masks_padding(self):
        generator = Generator(8).eval()
        generator.pair = RelativeMessages()
        generator.context = nn.Identity()
        target = torch.zeros(2, 8, 2)
        present = torch.tensor([[True, False, True], [False, False, False]])
        histories = torch.zeros(2, 3, 8, 2).masked_fill(~present[..., None, None], torch.nan)
        relative = torch.tensor([[[-2., -4.], [torch.nan, torch.nan], [-3., -1.]],
                                 [[torch.nan, torch.nan]] * 3])

        # Distinct neighbors supply the two maxima. Zero padding would
        # incorrectly dominate both negative coordinates in the first pool.
        with torch.no_grad():
            context = generator.encode(target, histories, relative, present)
            order = [2, 0, 1]
            shuffled = generator.encode(target, histories[:, order], relative[:, order], present[:, order])
        expected_pool = torch.tensor([[-2., -1.] * 8, [0., 0.] * 8])
        torch.testing.assert_close(context[:, 16:], expected_pool, rtol=0, atol=0)
        torch.testing.assert_close(shuffled, context, rtol=0, atol=0)


class FixedCandidates(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(4))
        self.register_buffer("paths", torch.tensor([[0.] * 11 + [1.], [1.25] * 12,
                                                   [2.5] * 12, [3.8] * 12]))

    def encode(self, target):
        return target[:, -1]

    def decode(self, context, target, latent):
        indices = latent[..., 0].long()
        x = self.paths[indices] + self.offset[indices][..., None]
        steps = x.diff(dim=-1, prepend=torch.zeros_like(x[..., :1]))
        return torch.stack((steps, torch.zeros_like(steps)), dim=-1)


class EndpointDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(()))

    def forward(self, history, future):
        return self.slope * (future[:, -1, 0] - history[:, -1, 0])


class VarietyTrainingTests(unittest.TestCase):
    def test_best_full_path_and_first_adversarial_draw_route_gradients(self):
        anchor = torch.tensor([0., 10.])
        history = torch.zeros(2, 8, 2)
        history[..., 0] = anchor[:, None]
        future = torch.zeros(2, 12, 2)
        future[..., 0] = anchor[:, None] + torch.tensor([[1.], [3.]])
        latent = torch.arange(4).reshape(1, 4, 1).expand(2, -1, -1)

        # Draw 0 reaches the first target's endpoint exactly, but its full
        # path loses to draw 1. Draw 2 wins for the second target.
        for weight in (0., 0.3):
            with self.subTest(adversarial_weight=weight):
                generator = FixedCandidates()
                discriminator = EndpointDiscriminator()
                g_optim = torch.optim.SGD(generator.parameters(), lr=0.)
                d_optim = torch.optim.SGD(discriminator.parameters(), lr=0.)
                losses = train_batch(generator, discriminator, g_optim, d_optim,
                                     (history,), future, latent, weight)
                real = torch.tensor([1., 3.])
                fake = torch.tensor(1.)
                expected_d_loss = (F.softplus(-real).mean() + F.softplus(fake)) / 2
                torch.testing.assert_close(losses, torch.stack((torch.tensor(0.375),
                                                                 F.softplus(-fake), expected_d_loss)))
                expected_g_grad = torch.tensor([0., 0.5, -0.5, 0.])
                expected_g_grad[0] = -weight * torch.sigmoid(-fake)
                torch.testing.assert_close(generator.offset.grad, expected_g_grad)
                expected_d_grad = ((-real * torch.sigmoid(-real)).mean() + fake * torch.sigmoid(fake)) / 2
                torch.testing.assert_close(discriminator.slope.grad, expected_d_grad)
                self.assertTrue(discriminator.slope.requires_grad)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
