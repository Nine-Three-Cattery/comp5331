"""Temporally coherent Gaussian anchor sampling, independent across pedestrians.

Stratified Gaussian anchors are shared across future steps for each trajectory.
This preserves each step's marginal Gaussian while avoiding independent noise
at every step. No target-dependent sample selection or variance shrinkage.
"""
import math
import numpy as np
import torch
from torch import nn
from .losses import gaussian_parameters


def anchor_sample(raw, count=20, generator=None):
    if count < 1:
        raise ValueError("count must be positive")
    b, _, n, _ = raw.shape
    # Separate random permutations for each scene, pedestrian and coordinate.
    ranks = torch.rand(count, b, n, 2, device=raw.device, generator=generator).argsort(0)
    uniform = (ranks + torch.rand(count, b, n, 2, device=raw.device, generator=generator)) / count
    z = math.sqrt(2) * torch.erfinv(2 * uniform.clamp(1e-6, 1 - 1e-6) - 1)
    z = z[:, :, None]
    mean, scale, rho = gaussian_parameters(raw)
    x = scale[..., 0] * z[..., 0]
    y = scale[..., 1] * (rho * z[..., 0] + (1 - rho.square()).sqrt() * z[..., 1])
    return mean + torch.stack((x, y), -1)


class GaussianMixtureAnchors(nn.Module):
    """Train-only GMM over normalized displacement-residual sequences.

    This is an explicit project reconstruction of GMM anchor sampling: the
    public artifact does not implement the paper's claimed GMM sampler.
    A component is chosen once for a whole future, preserving temporal modes.
    """
    def __init__(self, components=5, steps=12):
        super().__init__()
        self.register_buffer("weights", torch.ones(components) / components)
        self.register_buffer("means", torch.zeros(components, steps, 2))
        self.register_buffer("scales", torch.ones(components, steps, 2))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, dataset, seed=0, max_trajectories=50000):
        from sklearn.mixture import GaussianMixture
        residuals = []
        for ex in dataset:
            origin = ex["observations"][-1]
            previous = ex["observations"][-2]
            steps = np.diff(np.concatenate((origin[None], ex["future"])), axis=0)
            residuals.append((steps - (origin - previous)[None]).transpose(1, 0, 2))
        x = np.concatenate(residuals).reshape(-1, self.means.shape[1] * 2)
        if len(x) < len(self.weights):
            raise ValueError("Not enough training trajectories to fit anchor mixture")
        rng = np.random.default_rng(seed)
        if len(x) > max_trajectories:
            x = x[rng.choice(len(x), max_trajectories, replace=False)]
        x = (x - x.mean(0)) / x.std(0).clip(0.02)
        mixture = GaussianMixture(len(self.weights), covariance_type="diag", reg_covar=1e-3,
                                  random_state=seed, max_iter=200, n_init=1).fit(x)
        if not mixture.converged_:
            raise RuntimeError("Anchor GMM did not converge; no checkpoint was produced")
        # Exact mixture moment normalization, using training data only.
        center = np.sum(mixture.weights_[:, None] * mixture.means_, axis=0)
        variance = np.sum(mixture.weights_[:, None] * (mixture.covariances_ + (mixture.means_ - center)**2), axis=0)
        self.weights.copy_(torch.as_tensor(mixture.weights_))
        self.means.copy_(torch.as_tensor(((mixture.means_ - center) / variance**0.5).reshape(self.means.shape)))
        self.scales.copy_(torch.as_tensor((mixture.covariances_ / variance).reshape(self.means.shape)**0.5))
        self.fitted.fill_(True)
        return {"fit_trajectories": len(x), "iterations": mixture.n_iter_, "converged": bool(mixture.converged_)}

    def sample(self, raw, count=20, generator=None):
        if not self.fitted.item() or count < 1:
            raise ValueError("Fit the GMM on training data first and use count >= 1")
        b, t, n, _ = raw.shape
        # Stratify categorical component draws without choosing samples by target.
        ranks = torch.rand(count, b, n, device=raw.device, generator=generator).argsort(0)
        u = (ranks + torch.rand(count, b, n, device=raw.device, generator=generator)) / count
        components = torch.searchsorted(self.weights.cumsum(0), u.contiguous()).clamp_max(len(self.weights) - 1)
        noise = torch.randn(count, b, n, t, 2, device=raw.device, generator=generator)
        z = (self.means[components] + self.scales[components] * noise).permute(0, 1, 3, 2, 4)
        mean, scale, rho = gaussian_parameters(raw)
        x = scale[..., 0] * z[..., 0]
        y = scale[..., 1] * (rho * z[..., 0] + (1 - rho.square()).sqrt() * z[..., 1])
        return mean + torch.stack((x, y), -1)
