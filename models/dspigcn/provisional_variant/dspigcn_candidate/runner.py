"""Train/evaluate CLI. No imports from or writes into teammates' packages."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import io
import zipfile
import os
import platform
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DataConfig, SCENES, load_fold, collate
from .evaluation import CONDITIONS, metrics, motion, perturb, stable_seed, training_thresholds
from .losses import training_loss
from .model import DSPIGCN, ModelConfig
from .plotting import trajectory_plot
from .sampling import GaussianMixtureAnchors


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to export an empty results table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def seed_everything(seed, threads=1):
    # Explicit training/evaluation setup only; importing the inference interface
    # must not modify a teammate's process environment or random state.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def environment():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "platform": platform.platform()}


def default_config():
    return {"seeds": [0, 1, 2], "threads": 1, "anchor_components": 5,
            "data": asdict(DataConfig()), "model": asdict(ModelConfig()),
            "training": {"epochs": 100, "patience": 15, "batch_size": 32,
                         "learning_rate": .001, "reconstruction_weight": 1., "physics_weight": .1}}


def train(data_root, scene, seed, config, output, device):
    output = Path(output)
    if (output / "best.pt").exists():
        raise FileExistsError(f"Use a new run directory; checkpoint already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed, config.get("threads", 1))
    data_cfg = DataConfig(**config.get("data", {}))
    model_cfg = ModelConfig(**config.get("model", {}))
    if (data_cfg.obs_len, data_cfg.pred_len, data_cfg.dt) != (model_cfg.obs_len, model_cfg.pred_len, model_cfg.dt):
        raise ValueError("Data and model lengths/dt must agree")
    start = time.perf_counter()
    train_set = load_fold(data_root, scene, "train", data_cfg)
    val_set = load_fold(data_root, scene, "val", data_cfg)
    thresholds = training_thresholds(train_set, data_cfg.dt)
    for split, dataset in (("train", train_set), ("val", val_set)):
        save_json(output / f"manifest_{split}.json", dataset.manifest)
    preparation_s = time.perf_counter() - start
    anchors = GaussianMixtureAnchors(config.get("anchor_components", 5), model_cfg.pred_len)
    anchor_info = anchors.fit(train_set, seed)
    model = DSPIGCN(model_cfg).to(device)
    optim_cfg = config.get("training", {})
    epochs = optim_cfg.get("epochs", 100)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_cfg.get("learning_rate", 0.001), weight_decay=1e-4)
    batch_size = optim_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate)
    run = {"scene": scene, "seed": seed, "config": config, "environment": environment(),
           "preprocessing_seconds": preparation_s, "anchors": anchor_info,
           "parameters": sum(p.numel() for p in model.parameters()), "thresholds": thresholds,
           "protocol_status": "provisional_until_team_manifest_is_verified"}
    run["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(Path(__file__).parent.glob("*.py"))}
    save_json(output / "run.json", run)
    best, stale, history = float("inf"), 0, []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0., 0
        for batch in train_loader:
            obs, target, mask = (batch[k].to(device) for k in ("observations", "future", "mask"))
            optimizer.zero_grad(set_to_none=True)
            losses = training_loss(model(obs, mask), target, model_cfg.dt,
                                   optim_cfg.get("reconstruction_weight", 1.), optim_cfg.get("physics_weight", .1))
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            n = mask.sum().item()
            total += losses["total"].item() * n
            count += n
        model.eval()
        val_error, val_count = 0., 0
        with torch.inference_mode():
            for batch in val_loader:
                obs, target, mask = (batch[k].to(device) for k in ("observations", "future", "mask"))
                pred = model(obs, mask)["displacement"][..., :2].cumsum(1) + obs[:, -1:]
                error = torch.linalg.vector_norm(pred - target, dim=-1).mean(1)
                val_error += (error * mask).sum().item()
                val_count += mask.sum().item()
        val_ade = val_error / val_count
        if not np.isfinite(val_ade):
            raise FloatingPointError("Nonfinite validation ADE")
        history.append({"epoch": epoch, "train_loss": total / count, "val_mean_trajectory_ADE": val_ade})
        save_csv(output / "history.csv", history)
        print(json.dumps({"scene": scene, "seed": seed, **history[-1]}), flush=True)
        if val_ade < best:
            best, stale = val_ade, 0
            temporary = output / f"best.pt.{os.getpid()}.tmp"
            torch.save({"model": model.state_dict(), "model_config": asdict(model_cfg),
                        "data_config": asdict(data_cfg), "anchors": anchors.state_dict(),
                        "anchor_components": len(anchors.weights), "seed": seed, "scene": scene,
                        "epoch": epoch, "validation_ADE": best, "thresholds": thresholds,
                        "train_manifest_sha256": train_set.manifest["manifest_sha256"],
                        "val_manifest_sha256": val_set.manifest["manifest_sha256"]}, temporary)
            temporary.replace(output / "best.pt")
        else:
            stale += 1
        if stale >= optim_cfg.get("patience", 15):
            break
    sync(device)
    run.update(training_seconds=time.perf_counter() - start, completed_epochs=epoch, best_validation_ADE=best)
    save_json(output / "run.json", run)
    return output / "best.pt"


def load_checkpoint(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Parameter initialization is immediately replaced by checkpoint weights;
    # preserve the caller's CPU RNG instead of consuming its training seed.
    with torch.random.fork_rng(devices=[]):
        model = DSPIGCN(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    anchors = GaussianMixtureAnchors(state["anchor_components"], model.config.pred_len).to(device)
    anchors.load_state_dict(state["anchors"])
    return model, anchors, state


def evaluate(data_root, checkpoint, output, device, conditions=None, sample_count=20, corruption_seed=2026):
    output = Path(output)
    model, anchors, state = load_checkpoint(checkpoint, device)
    cfg = DataConfig(**state["data_config"])
    seed_everything(state["seed"])
    dataset = load_fold(data_root, state["scene"], "test", cfg)
    save_json(output / "manifest_test.json", dataset.manifest)
    test_id = dataset.manifest["manifest_sha256"]
    # Warm up with a fixed scene; sampling RNG is reset per scene below.
    with torch.inference_mode():
        warmup = torch.as_tensor(dataset[0]["observations"])[None].to(device)
        for _ in range(5):
            anchors.sample(model(warmup)["displacement"], sample_count)
    sync(device)
    summaries = []
    for condition in conditions or list(CONDITIONS):
        sigma, missing = CONDITIONS[condition]
        rows, close, pairs = [], 0, 0
        model_s, sampling_s = 0., 0.
        worst = []
        condition_dir = output / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(condition_dir / "predictions.zip", "w", allowZip64=True) as archive:
            for index, ex in enumerate(dataset):
                corrupted = perturb(ex["observations"], ex["ids"], sigma, missing, corruption_seed)
                obs = torch.as_tensor(corrupted)[None].to(device)
                generator = torch.Generator(device=device).manual_seed(stable_seed(ex["scene_id"], state["seed"]) % (2**63 - 1))
                with torch.inference_mode():
                    sync(device)
                    start = time.perf_counter()
                    result = model(obs)
                    sync(device)
                    model_s += time.perf_counter() - start
                    start = time.perf_counter()
                    samples = anchors.sample(result["displacement"], sample_count, generator).cumsum(2) + result["origin"][None, :, None]
                    sync(device)
                    sampling_s += time.perf_counter() - start
                    pred = samples[:, 0].cpu().numpy()
                accuracy = metrics(pred, ex["future"])
                diagnostics, pair_stats = motion(pred, corrupted, cfg.dt, state["thresholds"])
                close += pair_stats["close_pairs"]
                pairs += pair_stats["pair_timesteps"]
                for n, identity in enumerate(ex["ids"]):
                    rows.append({"sample_id": identity, **{key: float(value[n]) for key, value in {**accuracy, **diagnostics}.items()}})
                buffer = io.BytesIO()
                np.savez_compressed(buffer, observations=corrupted,
                                    clean_observations=ex["observations"], target=ex["future"], predictions=pred,
                                    sample_ids=np.asarray(ex["ids"]), frames=ex["frames"], pedestrians=ex["pedestrians"])
                archive.writestr(f"{index:06d}.npz", buffer.getvalue())
                score = float(accuracy["meanADE"].mean())
                worst.append((score, index, ex["scene_id"]))
                if index < 3:
                    trajectory_plot(corrupted, ex["future"], pred, condition_dir / f"example_{index}.png", ex["scene_id"] + " " + condition)
        save_csv(condition_dir / "per_trajectory.csv", rows)
        numeric = list(rows[0])[1:]
        summary = {"model": "DSPIGCN", "scene": state["scene"], "seed": state["seed"],
                   "condition": condition, "sample_count": sample_count, "test_set_id": test_id,
                   "trajectories": len(rows), "windows": len(dataset),
                   **{key: float(np.mean([r[key] for r in rows])) for key in numeric},
                   "close_approach_rate": close / pairs if pairs else 0., "pair_timesteps": pairs,
                   "prediction_seconds": model_s, "sampling_seconds": sampling_s}
        summaries.append(summary)
        failures = sorted(worst, reverse=True)[:3]
        save_json(condition_dir / "failures.json", [{"meanADE": score, "window_index": idx, "scene_id": identity} for score, idx, identity in failures])
        with zipfile.ZipFile(condition_dir / "predictions.zip") as archive:
            for _, idx, identity in failures:
                with np.load(io.BytesIO(archive.read(f"{idx:06d}.npz")), allow_pickle=False) as arrays:
                    trajectory_plot(arrays["observations"], arrays["target"], arrays["predictions"],
                                    condition_dir / f"failure_{idx}.png", identity + " " + condition)
        print(json.dumps(summary), flush=True)
    save_csv(output / "results.csv", summaries)
    save_json(output / "evaluation.json", {"checkpoint": str(Path(checkpoint).resolve()),
              "checkpoint_epoch": state["epoch"], "sample_count": sample_count, "corruption_seed": corruption_seed,
              "environment": environment(), "conditions": conditions or list(CONDITIONS),
              "timing": "5 warmups; synchronized model forward and sampling/integration separately; excludes IO/metrics/plots"})
    return summaries


def benchmark(checkpoint, output, device, pedestrians=32, repeats=100):
    model, anchors, _ = load_checkpoint(checkpoint, device)
    seed_everything(0)
    obs = torch.randn(1, model.config.obs_len, pedestrians, 2, device=device).cumsum(1)
    elapsed = {"preprocessing_ms": [], "prediction_ms": [], "sampling_ms": []}
    with torch.inference_mode():
        for i in range(repeats + 10):
            raw = obs.cpu().numpy()[0]
            start = time.perf_counter()
            data = perturb(raw, [str(j) for j in range(pedestrians)], sigma=.1)
            tensor = torch.as_tensor(data)[None].to(device)
            sync(device)
            middle = time.perf_counter()
            result = model(tensor)
            sync(device)
            end = time.perf_counter()
            anchors.sample(result["displacement"], 20).cumsum(2) + result["origin"][None, :, None]
            sync(device)
            finish = time.perf_counter()
            if i >= 10:
                for key, value in zip(elapsed, (middle - start, end - middle, finish - end)):
                    elapsed[key].append(1000 * value)
    report = {"pedestrians": pedestrians, "samples": 20, "warmups": 10, "repeats": repeats,
              "environment": environment(), "device": str(device),
              "boundaries": "preprocess = noise + host-to-device; prediction = derivatives/graph/model; sampling = GMM + integration",
              **{key: {"median": float(np.median(v)), "mean": float(np.mean(v)), "std": float(np.std(v))} for key, v in elapsed.items()}}
    save_json(output, report)
    return report


def run_matrix(data_root, config, output, device, resume=False):
    """Isolate phases in child processes; reuse only fully completed, matching runs.

    Process isolation releases native CUDA/plotting/filesystem resources between
    phases. It does not change model settings, data, seeds or optimizer steps.
    Incomplete training cannot resume from best-only weights: preserve that run
    separately and start its seed again rather than pretending optimizer state
    has been recovered.
    """
    root = Path(output).resolve()
    data_root = Path(data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "matrix_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing matrix configuration differs; use a new output directory")
    save_json(config_path, config)
    results = []
    for scene in SCENES:
        for seed in config.get("seeds", [0, 1, 2]):
            directory = root / scene / f"seed_{seed}"
            checkpoint = directory / "best.pt"
            run_path = directory / "run.json"
            if resume and checkpoint.is_file() and run_path.is_file():
                run = json.loads(run_path.read_text())
                if "completed_epochs" not in run:
                    raise ValueError(f"Incomplete training at {directory}; archive it and rerun this seed")
                if (run["scene"], run["seed"], run["config"]) != (scene, seed, config):
                    raise ValueError(f"Cannot reuse mismatched run at {directory}")
                for split in ("train", "val"):
                    current = load_fold(data_root, scene, split, DataConfig(**config.get("data", {}))).manifest
                    if current != json.loads((directory / f"manifest_{split}.json").read_text()):
                        raise ValueError(f"Cannot reuse changed source data at {directory}")
                print(f"Reusing completed training: {scene} seed {seed}", flush=True)
            else:
                subprocess.run([sys.executable, "-m", "dspigcn_candidate.runner", "train", "--data-root", str(data_root),
                                "--scene", scene, "--seed", str(seed), "--config", str(config_path),
                                "--output", str(directory), "--device", str(device)], check=True)
            evaluation = directory / "evaluation"
            table = evaluation / "results.csv"
            reuse_evaluation = resume and table.is_file() and (evaluation / "evaluation.json").is_file()
            if reuse_evaluation:
                with table.open() as f:
                    saved = list(csv.DictReader(f))
                current = load_fold(data_root, scene, "test", DataConfig(**config.get("data", {}))).manifest
                reuse_evaluation = (len(saved) == 6 and {r["condition"] for r in saved} == set(CONDITIONS)
                                    and all(r["test_set_id"] == current["manifest_sha256"] and int(r["sample_count"]) == 20
                                            and r["scene"] == scene and int(r["seed"]) == seed for r in saved)
                                    and all((evaluation / c / "predictions.zip").is_file() for c in CONDITIONS))
            if not reuse_evaluation:
                subprocess.run([sys.executable, "-m", "dspigcn_candidate.runner", "evaluate", "--data-root", str(data_root),
                                "--checkpoint", str(checkpoint), "--output", str(evaluation), "--device", str(device)], check=True)
            runtime_path = directory / "runtime.json"
            if not (resume and runtime_path.is_file()):
                subprocess.run([sys.executable, "-m", "dspigcn_candidate.runner", "benchmark", "--checkpoint", str(checkpoint),
                                "--output", str(runtime_path), "--device", str(device)], check=True)
            with table.open() as f:
                results.extend(csv.DictReader(f))
            save_csv(root / "results.csv", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate", "run-all", "prepare"):
        p = sub.add_parser(command)
        p.add_argument("--data-root", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        if command in {"train", "run-all", "prepare"}:
            p.add_argument("--config", help="JSON config (default: built-in five-fold / three-seed settings)")
        if command in {"train", "prepare"}:
            p.add_argument("--scene", choices=list(SCENES), default="eth")
        if command == "train":
            p.add_argument("--seed", type=int, default=0)
        if command == "run-all":
            p.add_argument("--resume", action="store_true", help="Reuse fully completed runs after verifying config and source manifests")
        if command == "evaluate":
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--conditions", nargs="+", choices=list(CONDITIONS))
            p.add_argument("--samples", type=int, default=20)
    p = sub.add_parser("benchmark")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pedestrians", type=int, default=32)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if getattr(args, "config", None) else default_config()
    if args.command == "train":
        train(args.data_root, args.scene, args.seed, config, args.output, args.device)
    elif args.command == "evaluate":
        evaluate(args.data_root, args.checkpoint, args.output, args.device, args.conditions, args.samples)
    elif args.command == "benchmark":
        benchmark(args.checkpoint, args.output, args.device, args.pedestrians)
    elif args.command == "prepare":
        for split in ("train", "val", "test"):
            dataset = load_fold(args.data_root, args.scene, split, DataConfig(**config.get("data", {})))
            save_json(Path(args.output) / f"manifest_{split}.json", dataset.manifest)
            print(split, len(dataset), dataset.manifest["trajectories"], dataset.manifest["manifest_sha256"])
    elif args.command == "run-all":
        run_matrix(args.data_root, config, args.output, args.device, args.resume)


if __name__ == "__main__":
    main()
