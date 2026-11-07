"""Train, export and time Social-GAN under the causal_v2 contract."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OWNER = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "li_junhao_trajectronpp"))

from independent_tpp.data import load_split
from independent_tpp.protocol import CONDITIONS, FOLDS, SEEDS, digest_json, sha256, write_json
from data import HistoryIndex, draw_scene_latent, scene_latents
from model import Discriminator, Generator, train_batch


DATA = ROOT / "data/causal_v2"
RUNTIME_WINDOWS = ROOT / "benchmark/fixtures/runtime_windows.json"


def bundle_path(run, tag):
    return run / "evaluation" / tag / f"predictions_test_{tag}.npz"


def index_split(fold, split, config):
    store = load_split(fold, split, DATA)
    return HistoryIndex(store, config["neighbor_radius_m"], config["neighbor_limit"])


def grouped(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def world_predictions(model, inputs, origins, latent):
    return model(*inputs, latent) + torch.from_numpy(origins)[:, None, None, :]


@torch.no_grad()
def validation_score(model, inputs, future, latent):
    energy = []
    ade = []
    for batch in grouped(np.arange(len(future)), 128):
        observed = tuple(value[batch] for value in inputs)
        samples = model(*observed, latent[batch])
        distances = (samples - future[batch, None]).norm(dim=-1)
        flat = samples.flatten(2)
        pair = torch.cdist(flat, flat, compute_mode="donot_use_mm_for_euclid_dist")
        count = samples.shape[1]
        energy.append(
            distances.square().mean(dim=-1).sqrt().mean(dim=-1)
            - pair.sum(dim=(1, 2)) / (2 * count * (count - 1) * (12 ** 0.5))
        )
        ade.append(distances.mean(dim=(1, 2)))
    return float(torch.cat(energy).mean()), float(torch.cat(ade).mean())


def train(fold, seed, profile, run, config):
    start = time.perf_counter()
    training = index_split(fold, "train", config)
    validation = index_split(fold, "val", config)
    config.update(train_identity=training.store.identity, val_identity=validation.store.identity)
    write_json(run / "config.json", config)
    train_count = len(training.observed)
    train_inputs, train_origins, _, _ = training.pack(np.arange(train_count))
    train_future = torch.from_numpy(training.future() - train_origins[:, None])
    val_inputs, val_origins, _, _ = validation.pack(np.arange(len(validation.observed)))
    val_future = torch.from_numpy(validation.future() - val_origins[:, None])
    val_latent, _ = scene_latents(validation, seed, config["sample_count"], config["latent_dim"])
    del training, validation
    generator = Generator(config["latent_dim"])
    discriminator = Discriminator()
    g_optim = torch.optim.Adam(generator.parameters(), lr=config["learning_rate"])
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate"])
    checkpoint = run / "selected.pt"
    candidates = []
    selected = None
    best_key = (float("inf"), float("inf"), float("inf"))
    epochs = config["pilot_epochs"] if profile == "pilot" else config["epochs"]
    for epoch in range(1, epochs + 1):
        generator.train()
        totals = torch.zeros(3)
        for batch in grouped(torch.randperm(train_count), config["batch_size"]):
            latent = torch.randn(len(batch), config["variety_draws"], config["latent_dim"])
            losses = train_batch(
                generator, discriminator, g_optim, d_optim,
                tuple(value[batch] for value in train_inputs), train_future[batch], latent,
                config["adversarial_weight"],
            )
            totals += losses * len(batch)
        generator.eval()
        energy, ade = validation_score(generator, val_inputs, val_future, val_latent)
        losses = (totals / train_count).tolist()
        row = dict(epoch=epoch, energy_score=energy, mean_ade=ade, variety_loss=losses[0],
                   generator_adversarial_loss=losses[1], discriminator_loss=losses[2])
        candidates.append(row)
        key = (energy, ade, epoch)
        if key < best_key:
            best_key = key
            selected = {**row, "checkpoint": checkpoint.name}
            pending = checkpoint.with_suffix(".tmp")
            torch.save(generator.state_dict(), pending)
            pending.replace(checkpoint)
        print(f"{profile} {fold}/{seed} epoch {epoch}/{epochs} val ES={energy:.5f} "
              f"meanADE={ade:.5f} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        if epoch - selected["epoch"] >= config["patience"]:
            break
    write_json(run / "checkpoint_selection.json", {
        "selection_split": "val", "test_used_for_selection": False,
        "criterion": ["energy_score", "mean_ade", "epoch"],
        "candidates": candidates, "selected": selected,
    })
    return checkpoint


@torch.inference_mode()
def export(model, test, run, seed, config, checkpoint):
    count = config["sample_count"]
    latent, sampling_seeds = scene_latents(test, seed, count, config["latent_dim"])
    labels = test.future()
    identity = {**test.store.identity, "seed": seed, "sample_count": count,
                "config_hash": digest_json(config), "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256(checkpoint)}
    size = len(labels)
    for tag, condition, level in CONDITIONS:
        samples = np.empty((size, count, 12, 2), dtype=np.float32)
        observed = np.empty((size, 8, 2), dtype=np.float32)
        mask = np.empty((size, 8), dtype=bool)
        for batch in grouped(np.arange(size), 128):
            inputs, origins, observed[batch], mask[batch] = test.pack(batch, seed, condition, level)
            samples[batch] = world_predictions(model, inputs, origins, latent[batch]).numpy()
        metadata = {**identity, "condition": condition, "level": level}
        records = []
        for row, sample_seed in zip(test.store.manifests, sampling_seeds):
            records.append(json.dumps({
                **{key: row.values[key] for key in
                   ("window_id", "scene_name", "scene_id", "source_file", "scene_t", "anchor_t")},
                "target_id": row.target_id, "fold": identity["fold"], "split": "test",
                "seed": seed, "sample_count": count, "sampling_seed": sample_seed,
                "dt": 0.4, "coordinate_unit": "meter", "condition": condition,
                "sigma": level if condition == "noise" else 0,
                "missing": level if condition == "missing" else 0,
            }))
        path = bundle_path(run, tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".tmp.npz")
        np.savez(pending, samples=samples, ground_truth=labels, observed=observed,
                 observed_mask=mask, records=np.asarray(records), metadata=json.dumps(metadata))
        pending.replace(path)


def environment(config):
    hardware = platform.processor()
    if platform.system() == "Darwin":
        details = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True)
        hardware = next((line.split(":", 1)[1].strip() for line in details.splitlines()
                         if "Chip:" in line), hardware)
    elif Path("/proc/cpuinfo").exists():
        hardware = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name")), hardware)
    return {"hardware": hardware, "device": "cpu", "threads": config["threads"],
            "system": platform.system(), "system_release": platform.release(),
            "architecture": platform.machine(), "python": platform.python_version(),
            "torch": str(torch.__version__), "numpy": np.__version__}


@torch.inference_mode()
def measure(fold, seed, run):
    config = json.loads((run / "config.json").read_text())
    torch.set_num_threads(config["threads"])
    model = load_model(config, run / "selected.pt")
    test = index_split(fold, "test", config)
    by_id = {row.window_id: i for i, row in enumerate(test.store.manifests)}
    ids = json.loads(RUNTIME_WINDOWS.read_text())[fold]
    batches = list(grouped(np.array([by_id[window_id] for window_id in ids]), 32))
    measurements = []
    for repetition in range(13):
        start = time.perf_counter()
        prepared = [test.pack(batch)[:2] for batch in batches]
        ready = time.perf_counter()
        # Generate exactly the draws required by the timing fixture.
        per_group = {}
        prediction = []
        for indices, (inputs, origins) in zip(batches, prepared):
            latents = []
            for i in indices:
                key = test.groups[i]
                if key not in per_group:
                    per_group[key] = draw_scene_latent(fold, "test", seed, key, config["sample_count"],
                                                       config["latent_dim"])[0]
                latents.append(per_group[key])
            prediction.append(world_predictions(model, inputs, origins,
                                                torch.from_numpy(np.stack(latents))))
        predicted = time.perf_counter()
        for batch in prediction:
            batch.numpy()
        end = time.perf_counter()
        if repetition >= 3:
            measurements.append({"preprocessing_ms": (ready - start) * 1000,
                                 "prediction_sampling_ms": (predicted - ready) * 1000,
                                 "end_to_end_ms": (end - start) * 1000})
    write_json(run / "runtime.json", {
        "fold": fold, "seed": seed, "windows": len(ids), "batch_size": 32,
        "sample_count": config["sample_count"], "warmups": 3,
        "environment": environment(config), "repetitions": measurements,
        "boundary": "In-memory observed histories to metric NumPy arrays. "
                    "One-time history indexing and I/O are excluded.",
    })


def load_model(config, checkpoint):
    model = Generator(config["latent_dim"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.eval()


def run_job(job):
    fold, seed, profile, run_root, resume = job
    config = json.loads((OWNER / "config.json").read_text())
    config.update(profile=profile, fold=fold, seed=seed)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    run = run_root / fold / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=resume)
    start = time.perf_counter()
    if resume and (run / "config.json").is_file():
        previous = json.loads((run / "config.json").read_text())
        if any(previous.get(key) != value for key, value in config.items()):
            raise ValueError(f"Training settings changed since {fold}/{seed} started")
    if resume and (run / "checkpoint_selection.json").is_file() and (run / "selected.pt").is_file():
        config = json.loads((run / "config.json").read_text())
        if (run / "execution.json").is_file() and all(
            bundle_path(run, tag).is_file() for tag, _, _ in CONDITIONS
        ):
            return
        checkpoint = run / "selected.pt"
    else:
        checkpoint = train(fold, seed, profile, run, config)
    model = load_model(config, checkpoint)
    test = index_split(fold, "test", config)
    export(model, test, run, seed, config, checkpoint)
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(run / "execution.json", {
        "elapsed_seconds": time.perf_counter() - start,
        "peak_rss_mib": usage / (1024 ** (2 if platform.system() == "Darwin" else 1)),
        "environment": environment(config),
    })
    print(f"MODEL_EXPORT_OK {fold}/{seed}", flush=True)


def replay(fold, seed, run):
    config = json.loads((run / "config.json").read_text())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    checkpoint = run / "selected.pt"
    model = load_model(config, checkpoint)
    test = index_split(fold, "test", config)
    with tempfile.TemporaryDirectory(prefix="social-gan-replay-") as scratch:
        for tag, _, _ in CONDITIONS:
            if not bundle_path(run, tag).is_file():
                raise FileNotFoundError(bundle_path(run, tag))
        export(model, test, Path(scratch), seed, config, checkpoint)
        for tag, _, _ in CONDITIONS:
            with np.load(bundle_path(run, tag)) as saved, np.load(bundle_path(Path(scratch), tag)) as fresh:
                for key in ("samples", "ground_truth", "observed", "observed_mask", "records", "metadata"):
                    np.testing.assert_array_equal(saved[key], fresh[key])
    print(f"MODEL_REPLAY_OK {fold}/{seed} all six bundles", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("pilot", "formal"))
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--timing-only", action="store_true")
    operation.add_argument("--replay", nargs=2, metavar=("FOLD", "SEED"),
                           help="recreate one run's bundles and check them against the stored arrays")
    args = parser.parse_args()
    root = args.runs or ROOT / "artifacts/social_gan" / args.profile / "runs"
    matrix = [("eth", 0)] if args.profile == "pilot" else [(fold, seed) for fold in FOLDS for seed in SEEDS]
    if args.replay:
        fold, seed_text = args.replay
        seed = int(seed_text)
        if (fold, seed) not in matrix:
            parser.error("replay must select a fold and seed in the profile")
        replay(fold, seed, root / fold / f"seed_{seed}")
        return
    if not args.timing_only:
        jobs = [(fold, seed, args.profile, root, args.resume) for fold, seed in matrix]
        if args.jobs == 1:
            for job in jobs:
                run_job(job)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                list(pool.map(run_job, jobs))
    # Keep timing isolated from training workers in this invocation.
    for fold, seed in matrix:
        measure(fold, seed, root / fold / f"seed_{seed}")


if __name__ == "__main__":
    main()
