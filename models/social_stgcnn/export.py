"""Write Social-STGCNN forecasts in the shared causal_v2 prediction format."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from li_junhao_trajectronpp.independent_tpp.protocol import (
    CONDITIONS, DT, OBSERVATION_POINTS, PREDICTION_HORIZON,
)
from .model import sample_world, scene_geometry
from .windows import GraphWindows


SAMPLE_COUNT = 20


def sampling_seed(training_seed: int, batch_ordinal: int) -> int:
    """All six conditions share the same draws for a given contiguous batch."""
    return 100003 * training_seed + batch_ordinal


def raw_for_batch(model, batch, device):
    """Run equal-width complete scenes and recover [B,12,5] in requested target order."""
    positions = batch["positions"].to(device)
    widths = batch["agent_mask"].sum(1).to(device)
    graph_index = batch["graph_index"].to(device)
    target_slot = batch["target_index"].to(device)
    target_width = widths[graph_index]
    selected = None
    for width in widths.unique().tolist():
        graphs = torch.where(widths == width)[0]
        # The release decoder reshapes the entire scene; padding cannot enter the network.
        increments, laplacian = scene_geometry(positions[graphs, :, :width].contiguous())
        raw = model.full_scene(increments, laplacian)
        targets = torch.where(target_width == width)[0]
        local_graph = torch.searchsorted(graphs, graph_index[targets])
        if selected is None:
            selected = raw.new_empty((len(graph_index), PREDICTION_HORIZON, 5))
        selected[targets] = raw[local_graph, :, target_slot[targets]]
    return selected


def target_history(batch):
    return batch["positions"][batch["graph_index"], :, batch["target_index"]]


@torch.inference_mode()
def sample_batch(model, batch, sampling_seed: int, device):
    """Sample twenty world-space paths on `device` with the model in evaluation mode."""
    raw = raw_for_batch(model, batch, device)
    generator = torch.Generator(device="cpu").manual_seed(sampling_seed)
    noise = torch.randn((len(raw), SAMPLE_COUNT, PREDICTION_HORIZON, 2),
                        dtype=raw.dtype, device="cpu", generator=generator)
    origin = target_history(batch)[:, -1].to(device=device, dtype=raw.dtype)
    return sample_world(raw, origin, noise.to(device))


def export_condition(model, dataset: GraphWindows, run_dir: Path, *, seed: int, tag: str,
                     batch_size: int, device, run_hashes: dict) -> Path:
    """Publish one manifest-ordered test bundle for the selected checkpoint."""
    if dataset.identity["split"] != "test":
        raise ValueError("Prediction export requires the test manifest")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    conditions = {name: (kind, level) for name, kind, level in CONDITIONS}
    condition, level = conditions[tag]
    metadata = {
        **dataset.identity,
        **{key: run_hashes[key] for key in ("config_hash", "checkpoint", "checkpoint_sha256")},
        "model_name": "social_stgcnn", "seed": seed, "sample_count": SAMPLE_COUNT,
        "condition": condition, "level": level, "batch_size": batch_size,
        "sampling_seed_policy": "100003 * training_seed + zero_based_contiguous_batch_index",
    }
    common = {
        "model_name": metadata["model_name"], "fold": dataset.identity["fold"], "split": "test",
        "seed": seed, "sample_count": SAMPLE_COUNT, "condition": condition, "dt": DT,
        "coordinate_unit": "meter", "sigma": level if condition == "noise" else 0.0,
        "missing": int(level) if condition == "missing" else 0,
    }
    count = len(dataset)
    samples = np.empty((count, SAMPLE_COUNT, PREDICTION_HORIZON, 2), dtype=np.float32)
    observed = np.empty((count, OBSERVATION_POINTS, 2), dtype=np.float32)
    masks = np.empty((count, OBSERVATION_POINTS), dtype=bool)
    records = []
    model.eval()
    for ordinal, start in enumerate(range(0, count, batch_size)):
        stop = min(start + batch_size, count)
        batch = dataset.batch(range(start, stop), condition, level, seed, include_future=False)
        batch_seed = sampling_seed(seed, ordinal)
        forecast = sample_batch(model, batch, batch_seed, device).cpu().numpy()
        if forecast.shape != samples[start:stop].shape or not np.isfinite(forecast).all():
            raise ValueError(f"Invalid forecasts in {tag} at manifest index {start}")
        samples[start:stop] = forecast
        observed[start:stop] = target_history(batch).numpy()
        masks[start:stop] = batch["observed_mask"].numpy()
        for row in dataset.records[start:stop]:
            record = {**row.values, **common, "target_id": row.target_id, "sampling_seed": batch_seed}
            records.append(json.dumps(record, sort_keys=True, separators=(",", ":")))

    output = Path(run_dir) / "evaluation" / tag / f"predictions_test_{tag}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        pending = Path(temporary) / output.name
        np.savez_compressed(pending, samples=samples, ground_truth=dataset.future, observed=observed,
                            observed_mask=masks, records=np.asarray(records), metadata=json.dumps(metadata, sort_keys=True))
        pending.replace(output)
    return output
