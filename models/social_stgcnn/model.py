"""Full-scene Social-STGCNN with the operators executed by the CVPR release."""

import math

import torch
from torch import Tensor, nn


OBSERVATION_STEPS = 8
FORECAST_STEPS = 12


def release_displacements(positions: Tensor, origin: Tensor) -> Tensor:
    """Apply the release's four-decimal world-coordinate transform before differencing.

    Supply the neutral coordinates in float64 to retain their source precision.
    The node increment features and supervised responses are float32.
    """
    sequence = torch.cat((origin[:, None], positions), dim=1).to(torch.float64)
    return torch.diff(torch.round(sequence, decimals=4), dim=1).to(torch.float32)


def scene_geometry(positions: Tensor) -> tuple[Tensor, Tensor]:
    """Return observed increments and normalized Laplacians for unpadded [B,8,A,2] scenes."""
    if positions.ndim != 4 or positions.shape[1] != OBSERVATION_STEPS or positions.shape[-1] != 2:
        raise ValueError("positions must have shape [batch, 8, real agents, 2]")
    increments = release_displacements(positions, positions[:, 0])
    difference = increments.unsqueeze(-2) - increments.unsqueeze(-3)
    squared_distance = difference.square().sum(dim=-1)
    distinct = squared_distance > 0
    # The release constructs its normalized Laplacian in double precision
    # after measuring separation between the input displacements.
    separation = squared_distance.double().masked_fill(~distinct, 1)
    weights = separation.rsqrt().masked_fill(~distinct, 0)
    weights = weights + torch.eye(positions.shape[2], device=positions.device, dtype=weights.dtype)
    degree = weights.sum(dim=-1)
    inverse_root = degree.rsqrt()
    laplacian = (torch.diag_embed(degree) - weights) * inverse_root.unsqueeze(-1) * inverse_root.unsqueeze(-2)
    return increments, laplacian.to(increments.dtype)


class SocialSTGCNN(nn.Module):
    """Run a single ST-GCNN and the five active TXP-CNN convolutions on complete scenes.

    Each batch member must have the same real number and consistent ordering
    of agents across its eight observations. Padded agents change the output
    because the decoder convolves across consecutive agent slots.
    """

    def __init__(self) -> None:
        super().__init__()
        channels = 5
        self.message_projection = nn.Conv2d(2, channels, 1)
        self.graph_norm = nn.BatchNorm2d(channels)
        self.graph_activation = nn.PReLU()
        self.temporal_filter = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
        self.temporal_norm = nn.BatchNorm2d(channels)
        self.bypass_projection = nn.Conv2d(2, channels, 1)
        self.bypass_norm = nn.BatchNorm2d(channels)
        self.block_activation = nn.PReLU()

        self.expand = nn.Conv2d(OBSERVATION_STEPS, FORECAST_STEPS, 3, padding=1)
        self.expand_activation = nn.PReLU()
        self.refinements = nn.ModuleList(
            nn.Conv2d(FORECAST_STEPS, FORECAST_STEPS, 3, padding=1) for _ in range(3)
        )
        self.refinement_activations = nn.ModuleList(nn.PReLU() for _ in self.refinements)
        self.readout = nn.Conv2d(FORECAST_STEPS, FORECAST_STEPS, 3, padding=1)

    def full_scene(self, increments: Tensor, laplacian: Tensor) -> Tensor:
        """Map [B,8,A,2] increments and [B,8,A,A] Laplacians to [B,12,A,5] parameters."""
        if increments.ndim != 4 or increments.shape[1] != OBSERVATION_STEPS or increments.shape[-1] != 2:
            raise ValueError("increments must have shape [batch, 8, real agents, 2]")
        batch, frames, agents, _ = increments.shape
        if laplacian.shape != (batch, frames, agents, agents):
            raise ValueError("laplacian must match the observed batch, frames and real agents")

        motion = increments.permute(0, 3, 1, 2)
        projected = self.message_projection(motion).transpose(1, 2)  # [B,8,5,A]
        messages = (projected @ laplacian).transpose(1, 2).contiguous()
        filtered = self.graph_activation(self.graph_norm(messages))
        filtered = self.temporal_norm(self.temporal_filter(filtered))
        bypass = self.bypass_norm(self.bypass_projection(motion))
        encoded = self.block_activation(filtered + bypass)

        # Both changes of shape in the released extrapolator reinterpret
        # contiguous storage. A time/feature transpose produces a different model.
        latent_width = encoded.shape[1]
        time_channels = encoded.contiguous().reshape(batch, frames, latent_width, agents)
        forecast = self.expand_activation(self.expand(time_channels))
        for convolution, activation in zip(self.refinements, self.refinement_activations):
            forecast = forecast + activation(convolution(forecast))
        forecast = self.readout(forecast)
        parameters = forecast.contiguous().reshape(batch, latent_width, FORECAST_STEPS, agents)
        return parameters.permute(0, 2, 3, 1)

    def forward(self, increments: Tensor, laplacian: Tensor) -> Tensor:
        return self.full_scene(increments, laplacian)


def future_displacements(future_world: Tensor, origin: Tensor) -> Tensor:
    """Form future increments in metres starting at the supplied final observation."""
    return torch.diff(future_world, dim=1, prepend=origin[:, None])


def _correlation(raw: Tensor) -> tuple[Tensor, Tensor]:
    logit = raw[..., 4]
    # For rho=tanh(logit), sqrt(1-rho^2) equals 1/cosh(logit).
    # The logarithmic form preserves this scale when tanh rounds to one.
    log_conditional_scale = math.log(2) - torch.logaddexp(logit, -logit)
    return logit.tanh(), log_conditional_scale


def bivariate_nll(raw: Tensor, target_steps: Tensor) -> Tensor:
    """Average the released Gaussian NLL, with its density floor of 1e-20."""
    correlation, log_conditional_scale = _correlation(raw)
    log_scales = raw[..., 2:4]
    normalized = (target_steps - raw[..., :2]) * (-log_scales).exp()
    first = normalized[..., 0]
    second = (normalized[..., 1] - correlation * first) * (-log_conditional_scale).exp()
    loss = (math.log(2 * math.pi) + log_scales.sum(dim=-1) + log_conditional_scale
            + (first.square() + second.square()) / 2)
    return loss.clamp_max(-math.log(1e-20)).mean()


def sample_world(raw: Tensor, origin: Tensor, noise: Tensor) -> Tensor:
    """Transform independent normals [B,K,12,2] into target paths in world metres."""
    correlation, log_conditional_scale = _correlation(raw)
    x = noise[..., 0]
    y = correlation[:, None] * x + log_conditional_scale[:, None].exp() * noise[..., 1]
    standard_steps = torch.stack((x, y), dim=-1)
    increments = raw[:, None, :, :2] + raw[:, None, :, 2:4].exp() * standard_steps
    return origin[:, None, None] + increments.cumsum(dim=2)
