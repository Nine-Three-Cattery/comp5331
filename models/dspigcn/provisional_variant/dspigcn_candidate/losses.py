"""Numerically stable Gaussian likelihood and metric polynomial constraints."""
import math
import torch


def gaussian_parameters(raw):
    return raw[..., :2], raw[..., 2:4].clamp(-5, 3).exp(), raw[..., 4].tanh() * 0.999


def gaussian_nll(raw, target):
    mean, scale, rho = gaussian_parameters(raw)
    z = (target - mean) / scale
    variance = 1 - rho.square()
    return (math.log(2 * math.pi) + scale.log().sum(-1) + 0.5 * variance.log()
            + (z.square().sum(-1) - 2 * rho * z[..., 0] * z[..., 1]) / (2 * variance))


def polynomial_motion(coefficients, steps, dt):
    """p(t)=sum c_j (t/horizon)^j; analytic dp/dt is measured in m/s."""
    horizon = steps * dt
    time = torch.arange(1, steps + 1, device=coefficients.device, dtype=coefficients.dtype) / steps
    order = torch.arange(1, coefficients.shape[-1] + 1, device=time.device, dtype=time.dtype)
    position = torch.einsum("bndj,tj->btnd", coefficients, time[:, None] ** order)
    velocity = torch.einsum("bndj,tj->btnd", coefficients,
                            order * time[:, None] ** (order - 1) / horizon)
    return position, velocity


def masked_mean(values, mask):
    weights = mask[:, None].expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def training_loss(output, future, dt=0.4, reconstruction_weight=1.0, physics_weight=0.1):
    origin, mask = output["origin"], output["mask"]
    increments = torch.diff(torch.cat((origin[:, None], future), 1), dim=1)
    displacement, velocity = output["displacement"], output["velocity"]
    nll = masked_mean(gaussian_nll(displacement, increments), mask)
    velocity_nll = masked_mean(gaussian_nll(velocity, increments / dt), mask)
    relative = displacement[..., :2].cumsum(1)
    reconstruction = masked_mean((relative + origin[:, None] - future).square().sum(-1), mask)
    poly_position, poly_velocity = polynomial_motion(output["coefficients"], future.shape[1], dt)
    physics = masked_mean((relative - poly_position).square().sum(-1)
                          + (velocity[..., :2] - poly_velocity).square().sum(-1)
                          + (displacement[..., :2] / dt - velocity[..., :2]).square().sum(-1), mask)
    total = nll + velocity_nll + reconstruction_weight * reconstruction + physics_weight * physics
    return {"total": total, "nll": nll, "velocity_nll": velocity_nll,
            "reconstruction": reconstruction, "physics": physics}
