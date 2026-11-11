"""Recurrent Social-GAN core for the Group 7 pedestrian benchmark."""

import torch
from torch import nn
from torch.nn import functional as F


def displacements(positions):
    return torch.diff(positions, dim=-2, prepend=positions[..., :1, :])


def to_world(displacement, anchor):
    return displacement.cumsum(dim=-2) + anchor[:, None, None, :]


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.encoder = nn.LSTM(16, 16, batch_first=True)
        self.pair = nn.Sequential(nn.Linear(18, 32), nn.ReLU(), nn.Linear(32, 16))
        self.context = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32 - latent_dim))
        self.decoder_embed = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.decoder = nn.LSTMCell(16, 32)
        self.output = nn.Linear(32, 2)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def encode(self, target, neighbors, relative, present):
        # Encode only real neighbors, even when a batch contains isolated targets.
        target_step = displacements(target)
        observed = torch.cat((target_step, displacements(neighbors[present])), dim=0)
        _, (hidden, _) = self.encoder(self.embed(observed))
        target_hidden, neighbor_hidden = hidden[0].split([len(target), len(observed) - len(target)])
        messages = self.pair(torch.cat((neighbor_hidden, relative[present]), dim=-1))
        padded = target_hidden.new_full((*present.shape, messages.shape[-1]), -torch.inf)
        padded[present] = messages
        pooled = padded.amax(dim=1)
        pooled = torch.where(present.any(dim=1, keepdim=True), pooled, 0.0)
        return self.context(torch.cat((target_hidden, pooled), dim=-1))

    def decode(self, context, target, latent):
        batch, samples, _ = latent.shape
        hidden = torch.cat((context[:, None, :].expand(-1, samples, -1), latent), dim=-1)
        hidden = hidden.flatten(0, 1)
        cell = torch.zeros_like(hidden)
        # A causal four-interval average provides a constant-velocity skip connection.
        baseline = ((target[:, -1] - target[:, -5]) / 4).repeat_interleave(samples, dim=0)
        step = (target[:, -1] - target[:, -2]).repeat_interleave(samples, dim=0)
        prediction = []
        for _ in range(12):
            hidden, cell = self.decoder(self.decoder_embed(step), (hidden, cell))
            step = baseline + self.output(hidden)
            prediction.append(step)
        return torch.stack(prediction, dim=1).reshape(batch, samples, 12, 2)

    def forward(self, target, neighbors, relative, present, latent):
        context = self.encode(target, neighbors, relative, present)
        return to_world(self.decode(context, target, latent), target[:, -1])


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(2, 16), nn.ReLU())
        self.encoder = nn.LSTM(16, 32, batch_first=True)
        self.logit = nn.Sequential(nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))

    def forward(self, history, future):
        joined = torch.cat((history, future), dim=1)
        _, (hidden, _) = self.encoder(self.embed(displacements(joined)))
        return self.logit(hidden[0]).squeeze(-1)


def train_batch(generator, discriminator, g_optim, d_optim, inputs, future, latent, weight):
    target = inputs[0]
    anchor = target[:, -1]
    context = generator.encode(*inputs)
    # The argmin is discrete. Recompute just the selected and adversarial draws
    # for backpropagation so the other candidates need no activation storage.
    with torch.no_grad():
        candidates = to_world(generator.decode(context, target, latent), anchor)
        winner = (candidates - future[:, None]).square().sum(dim=(-1, -2)).argmin(dim=1)
    d_optim.zero_grad(set_to_none=True)
    scores = discriminator(torch.cat((target, target)), torch.cat((future, candidates[:, 0])))
    real, fake = scores.chunk(2)
    d_loss = (F.softplus(-real).mean() + F.softplus(fake).mean()) / 2
    d_loss.backward()
    d_optim.step()

    g_optim.zero_grad(set_to_none=True)
    selected = latent[torch.arange(len(target)), winner]
    pair = torch.stack((selected, latent[:, 0]), dim=1)
    best, generated = to_world(generator.decode(context, target, pair), anchor).unbind(dim=1)
    variety = (best - future).flatten(1).norm(dim=1).mean() / (12 ** 0.5)
    discriminator.requires_grad_(False)
    adversarial = F.softplus(-discriminator(target, generated)).mean()
    g_loss = variety + weight * adversarial
    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0, error_if_nonfinite=True)
    g_optim.step()
    discriminator.requires_grad_(True)
    return torch.stack((variety.detach(), adversarial.detach(), d_loss.detach()))
