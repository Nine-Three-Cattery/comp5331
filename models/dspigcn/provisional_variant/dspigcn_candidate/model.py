"""Dual-stream temporal-attention graph model with polynomial physics coupling.

The architecture follows the DSPIGCN artifact's component layout, with explicit
metric units, batch-safe tensor layouts, and a derivative-constrained velocity
stream. See docs/IMPLEMENTATION.md for deviations from the released artifact.
Input: [batch, observed_time, pedestrians, xy], in meters.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    obs_len: int = 8
    pred_len: int = 12
    dt: float = 0.4
    hidden: int = 64
    heads: int = 4
    temporal_layers: int = 1
    graph_layers: int = 1
    prediction_layers: int = 5
    dropout: float = 0.1
    polynomial_order: int = 4


def differences(x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Backward difference; copy first available difference into the first slot."""
    diff = torch.diff(x, dim=1) / dt
    return torch.cat((diff[:, :1], diff), dim=1)


def graph_laplacian(displacement: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Artifact's inverse relative-motion-distance normalized graph Laplacian.

    Zero relative distances have zero off-diagonal affinity (as in seq_to_graph).
    Invalid padded pedestrians have no edges and do not affect normalization.
    """
    distance = torch.linalg.vector_norm(
        displacement.unsqueeze(3) - displacement.unsqueeze(2), dim=-1
    )
    n = displacement.shape[2]
    eye = torch.eye(n, device=displacement.device, dtype=displacement.dtype)
    valid = mask[:, None, :, None] & mask[:, None, None, :]
    affinity = torch.where(distance > 1e-6, distance.clamp_min(1e-6).reciprocal(), 0)
    affinity = (affinity + eye) * valid
    inv_degree = affinity.sum(-1).clamp_min(1e-6).rsqrt()
    normalized = affinity * inv_degree.unsqueeze(-1) * inv_degree.unsqueeze(-2)
    return (eye - normalized) * valid


class GraphBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.spatial = nn.Linear(hidden, hidden)
        self.temporal = nn.Conv2d(hidden, hidden, (3, 1), padding=(1, 0))
        self.norm = nn.LayerNorm(hidden)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency):
        residual = x
        x = torch.einsum("btij,btjh->btih", adjacency, self.spatial(x))
        x = self.temporal(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.activation(self.norm(residual + self.dropout(x)))


class Stream(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input = nn.Linear(2, cfg.hidden)
        layer = nn.TransformerEncoderLayer(
            cfg.hidden, cfg.heads, 2 * cfg.hidden, cfg.dropout,
            batch_first=True, activation="gelu"
        )
        self.attention = nn.TransformerEncoder(layer, cfg.temporal_layers,
                                               enable_nested_tensor=False)
        time = torch.arange(cfg.obs_len).unsqueeze(1)
        scale = torch.exp(torch.arange(0, cfg.hidden, 2) * (-math.log(10000) / cfg.hidden))
        encoding = torch.zeros(cfg.obs_len, cfg.hidden)
        encoding[:, 0::2] = torch.sin(time * scale)
        encoding[:, 1::2] = torch.cos(time * scale)
        self.register_buffer("encoding", encoding)
        self.graph = nn.ModuleList(GraphBlock(cfg.hidden, cfg.dropout)
                                   for _ in range(cfg.graph_layers))
        self.time_projection = nn.Linear(cfg.obs_len, cfg.pred_len)
        self.prediction = nn.ModuleList(
            nn.Conv2d(cfg.hidden, cfg.hidden, (3, 1), padding=(1, 0))
            for _ in range(cfg.prediction_layers)
        )
        self.head = nn.Linear(cfg.hidden, 5)
        self.act = nn.PReLU()

    def forward(self, x, adjacency):
        b, t, n, _ = x.shape
        x = self.input(x).permute(0, 2, 1, 3).reshape(b * n, t, -1)
        x = self.attention(x + self.encoding).reshape(b, n, t, -1).permute(0, 2, 1, 3)
        for block in self.graph:
            x = block(x, adjacency)
        x = self.time_projection(x.permute(0, 3, 2, 1)).permute(0, 1, 3, 2)
        for conv in self.prediction:
            x = x + self.act(conv(x))
        return self.head(x.permute(0, 2, 3, 1))


class DSPIGCN(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        if config.dt <= 0 or config.obs_len < 2 or config.pred_len < 1:
            raise ValueError("dt must be positive; obs_len >= 2 and pred_len >= 1")
        if config.hidden % config.heads or config.hidden % 2:
            raise ValueError("hidden must be even and divisible by heads")
        self.config = config
        self.displacement_stream = Stream(config)
        self.velocity_stream = Stream(config)
        self.physics = nn.Sequential(
            nn.Linear(10 * config.pred_len, config.hidden), nn.Tanh(),
            nn.Linear(config.hidden, config.hidden), nn.Tanh(),
            nn.Linear(config.hidden, 2 * config.polynomial_order)
        )

    def forward(self, observations, mask=None):
        if observations.ndim != 4 or observations.shape[1] != self.config.obs_len or observations.shape[-1] != 2:
            raise ValueError("observations must have shape [B, obs_len, N, 2]")
        if mask is None:
            mask = torch.ones(observations.shape[0], observations.shape[2],
                              dtype=torch.bool, device=observations.device)
        if mask.shape != (observations.shape[0], observations.shape[2]) or not mask.any(1).all():
            raise ValueError("mask must have shape [B, N] with at least one valid pedestrian per scene")
        displacement = differences(observations)
        adjacency = graph_laplacian(displacement, mask)
        dp = self.displacement_stream(displacement, adjacency)
        vp = self.velocity_stream(displacement / self.config.dt, adjacency)
        # Constant-velocity residual parameterization does not use future targets.
        dp = torch.cat((dp[..., :2] + displacement[:, -1:, :, :], dp[..., 2:]), -1)
        vp = torch.cat((vp[..., :2] + displacement[:, -1:, :, :] / self.config.dt, vp[..., 2:]), -1)
        joint = torch.cat((dp, vp), -1).permute(0, 2, 1, 3).flatten(2)
        coefficients = self.physics(joint).reshape(*joint.shape[:2], 2, self.config.polynomial_order)
        return {"displacement": dp, "velocity": vp, "coefficients": coefficients,
                "origin": observations[:, -1], "mask": mask}

    def predict(self, observations, num_samples=20, generator=None, mask=None):
        """Return absolute metric trajectories [K, B, T, N, 2]."""
        from .sampling import anchor_sample
        output = self(observations, mask)
        return anchor_sample(output["displacement"], num_samples, generator).cumsum(2) + output["origin"][None, :, None]
