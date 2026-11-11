"""Independent graph-CVAE pedestrian model used for the course pilot.

The implementation follows the Trajectron++ mechanism at the level needed by
the trajectory-only ETH/UCY experiment: history encoding, typed directed
neighbor interaction, a 25-way discrete latent variable, recurrent Gaussian
velocity decoding, and SingleIntegrator position integration.  It is written
from the paper-level equations and does not import the official repository.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over a set of edge embeddings."""

    def __init__(self, encoder_dim: int, query_dim: int, internal_dim: int = 32):
        super().__init__()
        self.encoder = nn.Linear(encoder_dim, internal_dim, bias=False)
        self.query = nn.Linear(query_dim, internal_dim, bias=False)
        self.score = nn.Linear(internal_dim, 1, bias=False)

    def forward(self, values: torch.Tensor, query: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # values: [B, M, D], query: [B, Q], mask: [B, M]
        if values.shape[1] == 0:
            return values.new_zeros((values.shape[0], values.shape[-1]))
        logits = self.score(
            torch.tanh(self.encoder(values) + self.query(query).unsqueeze(1))
        ).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        weights = weights * mask.to(weights.dtype)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weights = weights / denominator
        return torch.sum(values * weights.unsqueeze(-1), dim=1)


class BivariateGaussian:
    """Small tensor helper for correlated two-dimensional Gaussians."""

    @staticmethod
    def parameters(raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = raw[..., 0:2]
        std = F.softplus(raw[..., 2:4]) + 1e-3
        rho = torch.tanh(raw[..., 4]).clamp(-0.98, 0.98)
        return mean, std, rho

    @staticmethod
    def log_prob(
        value: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (value - mean) / std
        one_minus_rho2 = (1.0 - rho.square()).clamp_min(1e-5)
        quadratic = (
            normalized.square().sum(dim=-1)
            - 2.0 * rho * normalized[..., 0] * normalized[..., 1]
        ) / one_minus_rho2
        normalizer = (
            torch.log(torch.tensor(2.0 * torch.pi, device=value.device, dtype=value.dtype))
            + torch.log(std[..., 0])
            + torch.log(std[..., 1])
            + 0.5 * torch.log(one_minus_rho2)
        )
        return -0.5 * quadratic - normalizer

    @staticmethod
    def sample(
        mean: torch.Tensor,
        std: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        eps = torch.randn_like(mean)
        first = std[..., 0] * eps[..., 0]
        second = std[..., 1] * (
            rho * eps[..., 0] + torch.sqrt((1.0 - rho.square()).clamp_min(1e-5)) * eps[..., 1]
        )
        return mean + torch.stack([first, second], dim=-1)


class IndependentTrajectronPP(nn.Module):
    """Trajectory-only independent implementation of the pedestrian core."""

    def __init__(
        self,
        state_dim: int = 6,
        history_hidden: int = 32,
        edge_hidden: int = 32,
        latent_classes: int = 25,
        decoder_hidden: int = 128,
        horizon: int = 12,
        dt: float = 0.4,
        kl_weight: float = 100.0,
        kl_min: float = 0.07,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.history_hidden = history_hidden
        self.edge_hidden = edge_hidden
        self.latent_classes = latent_classes
        self.decoder_hidden = decoder_hidden
        self.horizon = horizon
        self.dt = dt
        self.kl_weight = kl_weight
        self.kl_min = kl_min

        self.history_encoder = nn.LSTM(state_dim, history_hidden, batch_first=True)
        self.edge_encoder = nn.LSTM(state_dim * 2, edge_hidden, batch_first=True)
        self.edge_attention = AdditiveAttention(edge_hidden, history_hidden, internal_dim=32)

        self.future_encoder = nn.LSTM(
            2, history_hidden, bidirectional=True, batch_first=True
        )
        context_dim = history_hidden + edge_hidden
        self.context_dim = context_dim
        self.prior_hidden = nn.Sequential(
            nn.Linear(context_dim, 32), nn.ReLU()
        )
        self.prior_logits = nn.Linear(32, latent_classes)
        self.posterior_logits = nn.Linear(context_dim + 2 * history_hidden, latent_classes)

        self.decoder_context = nn.Linear(context_dim + latent_classes, decoder_hidden)
        self.decoder_cell = nn.GRUCell(2 + latent_classes + context_dim, decoder_hidden)
        self.decoder_projection = nn.Linear(decoder_hidden, 5)

    def encode_history(
        self,
        target_history: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, (history_h, _) = self.history_encoder(target_history)
        history_embedding = history_h[-1]

        batch_size, neighbor_count, time_steps, state_dim = neighbors.shape
        if neighbor_count == 0:
            edge_context = target_history.new_zeros((batch_size, self.edge_hidden))
        else:
            # The pedestrian experiment has a single typed edge (PEDESTRIAN
            # -> PEDESTRIAN).  Following the reference sum state-combination
            # rule, aggregate neighbor states before the edge LSTM.  The
            # attention module is then applied over the one edge type; it is
            # intentionally a one-element attention set rather than a hidden
            # neighbor-level pooling rule.
            combined_neighbors = neighbors.sum(dim=1)
            edge_input = torch.cat([target_history, combined_neighbors], dim=-1)
            _, (edge_h, _) = self.edge_encoder(edge_input)
            edge_embedding = edge_h[-1]
            edge_present = neighbor_mask.any(dim=-1).any(dim=-1, keepdim=True)
            edge_values = edge_embedding.unsqueeze(1)
            edge_context = self.edge_attention(
                edge_values,
                history_embedding,
                edge_present,
            )
        return torch.cat([history_embedding, edge_context], dim=-1)

    def encode_future(self, future_position: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.future_encoder(future_position)
        # Bidirectional LSTM hidden order is forward, backward.
        return torch.cat([hidden[-2], hidden[-1]], dim=-1)

    def latent_logits(
        self,
        context: torch.Tensor,
        future_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        prior_logits = self.prior_logits(self.prior_hidden(context))
        if future_embedding is None:
            return prior_logits, None
        posterior_logits = self.posterior_logits(torch.cat([context, future_embedding], dim=-1))
        return prior_logits, posterior_logits

    def _decode_teacher(
        self,
        context: torch.Tensor,
        z: torch.Tensor,
        previous_velocity: torch.Tensor,
        future_velocity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]
        state = torch.tanh(self.decoder_context(torch.cat([context, z], dim=-1)))
        previous = previous_velocity
        means, stds, rhos = [], [], []
        decoder_condition = torch.cat([z, context], dim=-1)
        for time_index in range(self.horizon):
            decoder_input = torch.cat([previous, decoder_condition], dim=-1)
            state = self.decoder_cell(decoder_input, state)
            mean, std, rho = BivariateGaussian.parameters(self.decoder_projection(state))
            means.append(mean)
            stds.append(std)
            rhos.append(rho)
            previous = future_velocity[:, time_index]
        return torch.stack(means, dim=1), torch.stack(stds, dim=1), torch.stack(rhos, dim=1)

    def loss(self, batch: Dict[str, object], temperature: float = 0.5) -> Dict[str, torch.Tensor]:
        context = self.encode_history(
            batch["target_history"], batch["neighbors"], batch["neighbor_mask"]
        )
        future_embedding = self.encode_future(batch["future_position"])
        prior_logits, posterior_logits = self.latent_logits(context, future_embedding)
        q_probs = torch.softmax(posterior_logits, dim=-1)
        p_probs = torch.softmax(prior_logits, dim=-1)
        z = F.gumbel_softmax(posterior_logits, tau=temperature, hard=True, dim=-1)
        previous = batch["target_history"][:, -1, 2:4]
        means, stds, rhos = self._decode_teacher(
            context, z, previous, batch["future_velocity"]
        )
        log_prob = BivariateGaussian.log_prob(
            batch["future_velocity"], means, stds, rhos
        )
        reconstruction = -log_prob.mean()
        raw_kl = torch.sum(
            q_probs
            * (
                torch.log(q_probs.clamp_min(1e-8))
                - torch.log(p_probs.clamp_min(1e-8))
            ),
            dim=-1,
        ).mean()
        kl = raw_kl.clamp_min(self.kl_min)
        total = reconstruction + self.kl_weight * kl
        return {
            "loss": total,
            "reconstruction_nll": reconstruction.detach(),
            "kl": kl.detach(),
            "raw_kl": raw_kl.detach(),
            "prior_entropy": (-(p_probs * torch.log(p_probs.clamp_min(1e-8))).sum(-1)).mean().detach(),
        }

    def sample_positions(
        self,
        batch: Dict[str, object],
        num_samples: int = 20,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Generate metric-relative position samples with shape [B,K,T,2]."""

        context = self.encode_history(
            batch["target_history"], batch["neighbors"], batch["neighbor_mask"]
        )
        prior_logits, _ = self.latent_logits(context)
        probabilities = torch.softmax(prior_logits, dim=-1)
        batch_size = context.shape[0]
        if deterministic:
            categories = torch.argmax(probabilities, dim=-1).unsqueeze(0).expand(num_samples, -1)
        else:
            categories = torch.distributions.Categorical(probabilities).sample((num_samples,))
        z = F.one_hot(categories, num_classes=self.latent_classes).to(context.dtype)
        # [K, B, D] -> [K*B, D]
        expanded_context = context.unsqueeze(0).expand(num_samples, -1, -1).reshape(
            num_samples * batch_size, -1
        )
        z = z.reshape(num_samples * batch_size, -1)
        state = torch.tanh(self.decoder_context(torch.cat([expanded_context, z], dim=-1)))
        previous = batch["target_history"][:, -1, 2:4].unsqueeze(0).expand(
            num_samples, -1, -1
        ).reshape(num_samples * batch_size, 2)
        decoder_condition = torch.cat([z, expanded_context], dim=-1)
        velocities = []
        for _ in range(self.horizon):
            decoder_input = torch.cat([previous, decoder_condition], dim=-1)
            state = self.decoder_cell(decoder_input, state)
            mean, std, rho = BivariateGaussian.parameters(self.decoder_projection(state))
            if deterministic:
                previous = mean
            else:
                previous = BivariateGaussian.sample(mean, std, rho)
            velocities.append(previous)
        normalized_velocity = torch.stack(velocities, dim=1).reshape(
            num_samples, batch_size, self.horizon, 2
        )
        relative_positions = torch.cumsum(normalized_velocity * 2.0 * self.dt, dim=2)
        return relative_positions.permute(1, 0, 2, 3).contiguous()

    def forward(self, batch: Dict[str, object], temperature: float = 0.5) -> Dict[str, torch.Tensor]:
        return self.loss(batch, temperature=temperature)
