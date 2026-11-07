"""Target-window Social-STGCNN and its bivariate displacement distribution."""

import math

import torch
from torch import Tensor, nn


OBSERVATION_STEPS = 8
FORECAST_STEPS = 12


def normalized_adjacency(positions: Tensor, agent_mask: Tensor) -> Tensor:
    """Build one symmetric graph per frame. Padded rows and columns are zero."""
    valid = agent_mask[:, None, :]
    positions = positions.masked_fill(~valid[..., None], 0)
    separation = positions.unsqueeze(-2) - positions.unsqueeze(-3)
    distance = torch.linalg.vector_norm(separation, dim=-1)
    connected = valid.unsqueeze(-1) & valid.unsqueeze(-2) & (distance > 0)
    weights = distance.masked_fill(~connected, 1).reciprocal().masked_fill(~connected, 0)
    weights = weights + torch.diag_embed(agent_mask.to(positions.dtype))[:, None]
    # Real vertices have a self-loop, so only padding has degree below one.
    inverse_root = weights.sum(dim=-1).clamp_min(1).rsqrt()
    return weights * inverse_root.unsqueeze(-1) * inverse_root.unsqueeze(-2)


class SocialSTGCNN(nn.Module):
    """Map metric history [B,8,A,2] and a complete-history mask [B,A] to [B,12,5]."""

    def __init__(self) -> None:
        super().__init__()
        width = 5
        self.spatial = nn.Linear(2, width)
        self.spatial_activation = nn.PReLU()
        self.temporal = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.temporal_activation = nn.PReLU()

        # After the graph block, time becomes channels. Convolutions run
        # along feature coordinates; no kernel slides across pedestrian IDs.
        self.expand_time = nn.Sequential(
            nn.Conv1d(OBSERVATION_STEPS, FORECAST_STEPS, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.refine_time = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(FORECAST_STEPS, FORECAST_STEPS, kernel_size=3, padding=1),
                nn.PReLU(),
            )
            for _ in range(4)
        )
        self.distribution = nn.Linear(width, 5)

    @staticmethod
    def graph_messages(observed: Tensor, agent_mask: Tensor) -> Tensor:
        """Compute the full graph's target row before any learned operation."""
        if observed.ndim != 4 or observed.shape[1] != OBSERVATION_STEPS or observed.shape[-1] != 2:
            raise ValueError("observed must have shape [batch, 8, agents, 2]")
        if agent_mask.dtype != torch.bool or agent_mask.shape != (observed.shape[0], observed.shape[2]):
            raise ValueError("agent_mask must be a boolean [batch, agents] tensor")

        # Accumulate the geometric reduction in double precision before
        # rounding to the network dtype; graph width varies between batches.
        coordinates = observed.to(torch.float64)
        origin = coordinates[:, -1, 0]
        centered = (coordinates - origin[:, None, None]).masked_fill(~agent_mask[:, None, :, None], 0)
        adjacency = normalized_adjacency(centered, agent_mask)
        # A single graph layer and a nodewise decoder need only row zero,
        # but each neighbor's degree still comes from the full scene graph.
        return (adjacency[:, :, 0, :, None] * centered).sum(dim=2).to(observed.dtype)

    def from_messages(self, target_messages: Tensor) -> Tensor:
        """Apply every trainable layer to the eight target messages."""
        spatial = self.spatial_activation(self.spatial(target_messages))
        temporal = self.temporal_activation(self.temporal(spatial.transpose(1, 2)))
        hidden = self.expand_time(temporal.transpose(1, 2))
        for refine in self.refine_time:
            hidden = hidden + refine(hidden)
        return self.distribution(hidden)

    def forward(self, observed: Tensor, agent_mask: Tensor) -> Tensor:
        return self.from_messages(self.graph_messages(observed, agent_mask))


def future_displacements(future_world: Tensor, origin: Tensor) -> Tensor:
    """Return increments in metres. The first one starts at the visible target anchor."""
    return torch.diff(future_world, dim=1, prepend=origin[:, None])


def _gaussian_parameters(raw: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mean = raw[..., :2]
    log_scale = raw[..., 2:4].clamp(-5, 3)
    correlation = (1 - 1e-4) * raw[..., 4].tanh()
    conditional_scale = ((1 - correlation) * (1 + correlation)).sqrt()
    return mean, log_scale, correlation, conditional_scale


def bivariate_nll(raw: Tensor, target_steps: Tensor) -> Tensor:
    """Mean joint x/y negative log likelihood across targets and future frames."""
    mean, log_scale, correlation, conditional_scale = _gaussian_parameters(raw)
    standardized = (target_steps - mean) * (-log_scale).exp()
    x = standardized[..., 0]
    conditional_y = (standardized[..., 1] - correlation * x) / conditional_scale
    return (
        math.log(2 * math.pi)
        + log_scale.sum(dim=-1)
        + conditional_scale.log()
        + 0.5 * (x.square() + conditional_y.square())
    ).mean()


def sample_world(raw: Tensor, origin: Tensor, noise: Tensor) -> Tensor:
    """Turn independent standard normals [B,K,12,2] into metric target paths."""
    mean, log_scale, correlation, conditional_scale = _gaussian_parameters(raw)
    x = noise[..., 0]
    y = correlation[:, None] * x + conditional_scale[:, None] * noise[..., 1]
    standard_steps = torch.stack((x, y), dim=-1)
    increments = mean[:, None] + log_scale[:, None].exp() * standard_steps
    return origin[:, None, None] + increments.cumsum(dim=2)
