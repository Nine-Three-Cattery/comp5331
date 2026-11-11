"""Independent implementation of Guo et al., KDD 2026, Eqs. 1--12.

Inputs: metric histories [B,H,N,2], boolean node mask [B,N]. Node 0
is the target. No label or future tensor is accepted by forward().
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

DT, H, T = 0.4, 8, 12


def normalized_graph(positions, mask):
    distance = torch.linalg.vector_norm(positions.unsqueeze(3) - positions.unsqueeze(2), dim=-1)
    active = mask[:, None, :, None] & mask[:, None, None, :]
    # Coincident nodes have zero off-diagonal weight (paper Eq. 1).
    adjacency = torch.where((distance > 1e-6) & active, distance.clamp_min(1e-6).reciprocal(), 0.)
    adjacency = adjacency + torch.diag_embed(mask.to(positions.dtype))[:, None]
    degree = adjacency.sum(-1).clamp_min(1e-12).rsqrt()
    return adjacency * degree.unsqueeze(-1) * degree.unsqueeze(-2)


def motion_inputs(positions):
    displacement = torch.zeros_like(positions)
    displacement[:, 1:] = positions[:, 1:] - positions[:, :-1]
    return displacement, displacement / DT


def polynomial(coefficients, times):
    """P(t) and analytic dP/dt in meters and m/s; coefficients use seconds."""
    powers = torch.arange(coefficients.shape[-2], device=times.device, dtype=times.dtype)
    basis = times[:, None] ** powers
    derivative = powers * times[:, None] ** (powers - 1).clamp_min(0)
    return (torch.einsum('tk,bkc->btc', basis, coefficients),
            torch.einsum('tk,bkc->btc', derivative, coefficients))


class Stream(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, width))
        self.attention = nn.MultiheadAttention(width, 4, dropout=dropout, batch_first=True)
        self.graph = nn.Linear(width, width)
        self.temporal = nn.Conv1d(width, width, kernel_size=3)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, features, adjacency, mask):
        b, h, n, _ = features.shape
        x = self.embed(features).permute(0, 2, 1, 3).reshape(b*n, h, -1)
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm(x + attended).reshape(b, n, h, -1).permute(0, 2, 1, 3)
        x = F.relu(torch.einsum('btij,btjc->btic', adjacency, self.graph(x)))
        x = x.permute(0, 2, 3, 1).reshape(b*n, -1, h)
        x = F.relu(x + self.dropout(self.temporal(F.pad(x, (2, 0)))))
        return x.reshape(b, n, -1, h)[:, 0].flatten(1)


def decoder(inputs, width, outputs):
    # Four affine layers, including the output layer.
    return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(), nn.Linear(width, outputs))


class DSPIGCN(nn.Module):
    def __init__(self, width=64, dropout=0.1):
        super().__init__()
        if width % 4:
            raise ValueError('width must be divisible by four attention heads')
        self.displacement = Stream(width, dropout)
        self.velocity = Stream(width, dropout)
        self.position_head = decoder(H*width, width, T*5)
        self.velocity_head = decoder(H*width, width, T*2)
        self.physics = nn.Sequential(nn.Conv1d(5, 64, 3, padding=1), nn.Tanh(),
                                     nn.Conv1d(64, 64, 3, padding=1), nn.Tanh(),
                                     nn.Conv1d(64, 64, 3, padding=1), nn.Tanh(),
                                     nn.Conv1d(64, 10, T))
        self.register_buffer('times', torch.arange(1, T+1)*DT)
        # Residual parameterization around observed constant velocity. Documented
        # numerical initialization, with all branches and losses still trainable.
        nn.init.zeros_(self.position_head[-1].weight)
        nn.init.zeros_(self.position_head[-1].bias)
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)
        nn.init.zeros_(self.physics[-1].weight)
        nn.init.zeros_(self.physics[-1].bias)

    def forward(self, positions, mask):
        if positions.ndim != 4 or positions.shape[1] != H or positions.shape[-1] != 2:
            raise ValueError('expected [B,8,N,2] histories')
        if mask.shape != (positions.shape[0], positions.shape[2]) or not mask[:, 0].all():
            raise ValueError('every graph must contain target node 0')
        d, v = motion_inputs(positions)
        a = normalized_graph(positions, mask)
        raw = self.position_head(self.displacement(d, a, mask)).reshape(-1, T, 5)
        last_velocity = v[:, -1, 0]
        mean = raw[..., :2] + self.times[None, :, None]*last_velocity[:, None]
        log_scale = raw[..., 2:4].clamp(-5., 3.)
        rho = 0.999*raw[..., 4].tanh()
        velocity = self.velocity_head(self.velocity(v, a, mask)).reshape(-1, T, 2) + last_velocity[:, None]
        time = self.times[None, :, None].expand(len(mean), -1, -1)
        scaled_coeff = self.physics(torch.cat((mean, velocity, time), -1).transpose(1, 2)).reshape(-1, 5, 2)
        # Predict coefficients on normalized time, then convert to seconds before
        # evaluating Eq. 8. This keeps t^4 conditioning manageable without losing dt.
        scales = (T*DT)**torch.arange(5, device=mean.device, dtype=mean.dtype)
        coeff = scaled_coeff / scales[None, :, None]
        coeff = coeff + torch.stack((torch.zeros_like(last_velocity), last_velocity,
                                    *[torch.zeros_like(last_velocity) for _ in range(3)]), 1)
        physical_position, physical_velocity = polynomial(coeff, self.times)
        return dict(mean=mean, log_scale=log_scale, rho=rho, velocity=velocity,
                    coefficients=coeff, physical_position=physical_position,
                    physical_velocity=physical_velocity)


def objective(output, target, physics_weight=0.1, reconstruction_weight=0.01):
    """Eq. 11, averaged over targets and time; Eq. 10 is squared L2."""
    error = (target-output['mean']) * (-output['log_scale']).exp()
    rho = output['rho']
    remaining = 1-rho.square()
    quadratic = (error.square().sum(-1)-2*rho*error[..., 0]*error[..., 1])/remaining
    nll = (math.log(2*math.pi)+output['log_scale'].sum(-1)+0.5*remaining.log()+0.5*quadratic).mean()
    reconstruction = (target-output['mean']).square().sum(-1).mean()
    physical = ((output['physical_position']-output['mean']).square().sum(-1).mean()
                +(output['physical_velocity']-output['velocity']).square().sum(-1).mean())
    loss = nll + physics_weight*physical + reconstruction_weight*reconstruction
    return loss, dict(nll=nll, reconstruction=reconstruction, physics=physical)


def anchor_sample(output, seeds, count=20):
    """Paper Algorithm 2: pooled time-mixture anchors, nearest unused per time.

    Stratified equal-weight mixture: exactly K anchors from each of the T
    components. All K*T anchors are used once, with no target access. Seeded
    noise is generated on CPU to be invariant to batching and device.
    """
    if count < 1 or len(seeds) != len(output['mean']):
        raise ValueError('positive sample count and one seed per target required')
    mean, scale, rho = output['mean'], output['log_scale'].exp(), output['rho']
    noise = torch.stack([torch.randn(T, count, 2, generator=torch.Generator().manual_seed(int(seed))) for seed in seeds]).to(mean)
    x = noise[..., 0]
    y = rho[..., None]*x+(1-rho.square()).sqrt()[..., None]*noise[..., 1]
    anchors = mean[:, :, None]+scale[:, :, None]*torch.stack((x, y), -1)
    anchors = anchors.flatten(1, 2)
    available = torch.ones(anchors.shape[:2], dtype=torch.bool, device=mean.device)
    distances = (mean[:, :, None]-anchors[:, None]).square().sum(-1)
    rows = torch.arange(len(mean), device=mean.device)
    paths = []
    for _ in range(count):
        path = []
        for t in range(T):
            chosen = distances[:, t].masked_fill(~available, float('inf')).argmin(-1)
            path.append(anchors[rows, chosen])
            available[rows, chosen] = False
        paths.append(torch.stack(path, 1))
    return torch.stack(paths, 1)
