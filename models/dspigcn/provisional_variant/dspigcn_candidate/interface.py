"""Explicit bridge for combining independently implemented trajectory models.

Uses the common time-major [T, N, 2] convention and optional Social-GAN-style
seq_start_end boundaries. Each scene is processed separately to prevent edges
between unrelated scenes. Requires no modifications in another model package.
"""
import torch

from .runner import load_checkpoint


class Predictor:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.model, self.anchors, self.metadata = load_checkpoint(checkpoint, self.device)

    @torch.inference_mode()
    def predict(self, obs_traj, num_samples=20, seq_start_end=None, seed=0):
        """Metric absolute positions in; [K, 12, N, 2] torch tensor out.

        Observed tensors are copied to the configured device. For interoperability
        the returned tensor is placed on the input tensor's device, or CPU when
        the input is a NumPy array. No in-place changes to the caller's input.
        """
        destination = obs_traj.device if isinstance(obs_traj, torch.Tensor) else torch.device("cpu")
        observations = torch.as_tensor(obs_traj, dtype=torch.float32, device=self.device)
        if observations.ndim != 3 or observations.shape[0] != self.model.config.obs_len or observations.shape[-1] != 2:
            raise ValueError("Expected observations [8, N, 2] in meters")
        if observations.shape[1] < 1 or not torch.isfinite(observations).all():
            raise ValueError("Observations must contain finite trajectories")
        boundaries = [(0, observations.shape[1])] if seq_start_end is None else torch.as_tensor(seq_start_end).cpu().tolist()
        cursor = 0
        for start, end in boundaries:
            if int(start) != start or int(end) != end or start != cursor or end <= start or end > observations.shape[1]:
                raise ValueError("seq_start_end must partition all pedestrians contiguously")
            cursor = end
        if cursor != observations.shape[1]:
            raise ValueError("seq_start_end must cover every pedestrian")
        results = []
        for scene, (start, end) in enumerate(boundaries):
            output = self.model(observations[None, :, int(start):int(end)])
            generator = torch.Generator(device=self.device).manual_seed(seed + scene)
            displacement = self.anchors.sample(output["displacement"], num_samples, generator)
            prediction = displacement.cumsum(2) + output["origin"][None, :, None]
            results.append(prediction[:, 0])
        return torch.cat(results, 2).to(destination)
