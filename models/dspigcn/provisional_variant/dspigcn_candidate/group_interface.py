"""Inference bridge for the group's causal GraphDataset/collate batches.

This changes tensor layout and restores metric origins only. It does not make
the candidate's training split or saved experiments causal_v2 compliant.
"""
import torch

from .interface import Predictor


class GroupPredictor(Predictor):
    @torch.inference_mode()
    def predict(self, observation_batch, num_samples=20, seed=0):
        """Return CPU [N,K,12,2]; target is the first valid graph node.

        Accept positions [N,8,A,2], mask [N,A], origin [N,2], and stable
        manifest indices. Only these visible-history fields enter inference.
        Ground truth and stored derivatives are never read. Sampling seeds
        follow the existing owner adapter's seed + 1009*manifest_index rule.
        """
        positions = torch.as_tensor(observation_batch["positions"])
        mask = torch.as_tensor(observation_batch["mask"], device=positions.device)
        origin = torch.as_tensor(observation_batch["origin"], device=positions.device)
        indices = list(observation_batch["indices"])
        if (positions.ndim != 4 or positions.shape[1] != 8 or positions.shape[-1] != 2
                or positions.shape[0] < 1 or positions.shape[2] < 1):
            raise ValueError("Expected nonempty positions [N,8,A,2]")
        if (mask.dtype != torch.bool or mask.shape != (positions.shape[0], positions.shape[2])
                or origin.shape != (positions.shape[0], 2) or not mask[:, 0].all()
                or len(indices) != positions.shape[0]):
            raise ValueError("Invalid mask, origins, or manifest indices")
        if any(int(index) != index or index < 0 for index in indices):
            raise ValueError("Manifest indices must be nonnegative integers")
        predictions = []
        for i, index in enumerate(indices):
            visible = positions[i, :, mask[i]] + origin[i]
            samples = super().predict(visible, num_samples=num_samples,
                                      seed=int(seed) + 1009 * int(index))
            predictions.append(samples[:, :, 0].cpu())
        return torch.stack(predictions)
