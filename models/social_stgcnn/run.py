"""Train, select, export and time one author-owned causal_v2 run."""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from li_junhao_trajectronpp.independent_tpp.protocol import CONDITIONS, FOLDS, SEEDS, digest_json, sha256, write_json
from li_junhao_trajectronpp.independent_tpp.data import load_split
from models.social_stgcnn.export import export_condition, sample_batch
from models.social_stgcnn.model import SocialSTGCNN, bivariate_nll, future_displacements
from models.social_stgcnn.timing import timing_batch
from models.social_stgcnn.windows import GraphWindows

SETTINGS = HERE / "configs/experiment.json"
DEFAULT_DATA = ROOT / "data/causal_v2"
ARTIFACTS = ROOT / "artifacts/social_stgcnn"


def environment(device):
    gpu = None
    driver = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        driver = (probe.stdout.splitlines() or [None])[0]
    processor = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        processor = next((line.partition(":")[2].strip()
                          for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), processor)
    return {"os": platform.system(), "host": platform.node(), "architecture": platform.machine(), "cpu": processor,
            "threads": torch.get_num_threads(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": str(torch.__version__), "device": str(device),
            "gpu": gpu, "cuda": torch.version.cuda, "driver": driver}


def dense_clean(windows, device):
    """Cache the exact parameter-free target messages for the clean split."""
    messages, steps = [], []
    for start in range(0, len(windows), 64):
        batch = windows.batch(np.arange(start, min(start + 64, len(windows))))
        pos = batch["positions"]
        messages.append(SocialSTGCNN.graph_messages(pos, batch["agent_mask"]))
        steps.append(future_displacements(batch["future"], pos[:, -1, 0]))
    return torch.cat(messages).to(device), torch.cat(steps).to(device)


@torch.inference_mode()
def validation_nll(model, split, batch_size):
    messages, steps = split
    total = torch.zeros((), dtype=torch.float64, device=messages.device)
    model.eval()
    for start in range(0, len(messages), batch_size):
        batch = slice(start, start + batch_size)
        total += bivariate_nll(model.from_messages(messages[batch]), steps[batch]).double() * len(steps[batch])
    value = total.item() / len(messages)
    if not np.isfinite(value):
        raise FloatingPointError("Non-finite validation NLL")
    return value


def fit(run, stage, fold, seed, data_root, device):
    settings = json.loads(SETTINGS.read_text())
    neighbors = settings["max_neighbors"]
    train = GraphWindows.from_split(fold, "train", data_root, max_neighbors=neighbors)
    validation = GraphWindows.from_split(fold, "val", data_root, max_neighbors=neighbors)
    epochs = settings[f"{stage}_epochs"]
    config = {**settings, "fold": fold, "seed": seed, "stage": stage, "epochs": epochs,
              "train_identity": train.identity, "val_identity": validation.identity,
              "environment": environment(device),
              "source_sha256": {n: sha256(HERE / n) for n in
                                ("model.py", "windows.py", "export.py", "run.py", "timing.py", "configs/experiment.json")}}
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    x, y = dense_clean(train, device)
    val = dense_clean(validation, device)
    del train, validation
    torch.manual_seed(seed)
    model = SocialSTGCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=settings["learning_rate"])
    shuffle = torch.Generator(device="cpu").manual_seed(seed)
    best = float("inf")
    selected = None
    candidates = []
    batch_size = settings["batch_size"]
    started = time.monotonic()
    with (run / "training.csv").open("x", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "train_nll", "validation_nll", "learning_rate", "elapsed_s"])
        for epoch in range(1, epochs + 1):
            if epoch == 151:
                optimizer.param_groups[0]["lr"] = settings["learning_rate_after_epoch_150"]
            model.train()
            order = torch.randperm(len(x), generator=shuffle).to(device)
            accumulated = torch.zeros((), dtype=torch.float64, device=device)
            for ix in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = bivariate_nll(model.from_messages(x[ix]), y[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"],
                                               error_if_nonfinite=True)
                optimizer.step()
                accumulated += loss.detach().double() * len(ix)
            train_nll = accumulated.item() / len(x)
            val_nll = None
            if epoch == 1 or epoch == epochs or epoch % settings["validation_interval"] == 0:
                val_nll = validation_nll(model, val, batch_size)
                candidates.append({"epoch": epoch, "validation_nll": val_nll})
                if val_nll < best:
                    best = val_nll
                    selected = {"checkpoint": "best.pt", "epoch": epoch, "validation_nll": best}
                    pending = run / "best.pt.tmp"
                    torch.save(model.state_dict(), pending)
                    pending.replace(run / "best.pt")
                print(f"{stage} {fold} seed={seed} epoch={epoch}/{epochs} "
                      f"train_nll={train_nll:.4f} val_nll={val_nll:.4f}", flush=True)
            writer.writerow([epoch, train_nll, val_nll, optimizer.param_groups[0]["lr"],
                             time.monotonic() - started])
            log.flush()
    write_json(run / "checkpoint_selection.json", {
        "selection_split": "val", "test_used_for_selection": False, "rule": settings["selection"],
        "selected": selected, "candidates": candidates, "training_seconds": time.monotonic() - started,
    })


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_batch_sequence(model, store, records, device):
    measurements = []
    for rep in range(13):
        synchronize(device)
        begin = time.perf_counter()
        batches = []
        for offset in range(0, len(records), 32):
            source = timing_batch(store, records[offset:offset + 32])
            batches.append({key: source[key].to(device) for key in ("positions", "agent_mask")})
        synchronize(device)
        prepared = time.perf_counter()
        predictions = [sample_batch(model, batch, 1009 * rep + i, device) for i, batch in enumerate(batches)]
        synchronize(device)
        predicted = time.perf_counter()
        cpu_outputs = [p.cpu() for p in predictions]
        synchronize(device)
        completed = time.perf_counter()
        if rep >= 3:
            measurements.append({"preprocessing_ms": (prepared-begin)*1000,
                                 "prediction_sampling_ms": (predicted-prepared)*1000,
                                 "end_to_end_ms": (completed-begin)*1000})
        del cpu_outputs, predictions
    return {
        "window_ids": [r.window_id for r in records], "targets": len(records),
        "batch_size": 32, "sample_count": 20, "warmups": 3, "measured_repetitions": 10,
        "repetitions": measurements,
        "medians": {k: statistics.median(r[k] for r in measurements) for k in measurements[0]},
    }


def time_model(run, model, store, dataset, fold, device, by_density):
    lookup = {record.window_id: record for record in dataset.records}
    ids = json.loads((ROOT / "benchmark/fixtures/runtime_windows.json").read_text())[fold]
    result = measure_batch_sequence(model, store, [lookup[uid] for uid in ids], device)
    result["environment"] = environment(device)
    result["timing_source_sha256"] = sha256(HERE / "timing.py")
    result["boundary"] = "In-memory neutral SceneStore and ordered target IDs through discovery of visible peers, graph-window assembly, prediction, sampling and CPU readback."
    if by_density:
        population = dataset.available_neighbors
        result["density"] = []
        for label, selected in (
            ("0", population == 0), ("1-3", (population >= 1) & (population <= 3)),
            ("4-7", (population >= 4) & (population <= 7)), ("8+", population >= 8),
        ):
            indices = np.flatnonzero(selected)
            row = {"available_neighbors_bin": label, "available_windows": len(indices), "timed_windows": 0}
            if len(indices) >= 32:
                row.update(measure_batch_sequence(model, store, [dataset.records[i] for i in indices[:32]], device))
                row["timed_windows"] = 32
            result["density"].append(row)
    write_json(run / "runtime.json", result)


def restore(run, device):
    config = json.loads((run / "config.json").read_text())
    selection = json.loads((run / "checkpoint_selection.json").read_text())
    if selection["selection_split"] != "val" or selection["test_used_for_selection"]:
        raise ValueError("Checkpoint selection must use validation only")
    checkpoint = selection["selected"]["checkpoint"]
    model = SocialSTGCNN().to(device)
    model.load_state_dict(torch.load(run / checkpoint, map_location=device, weights_only=True))
    model.eval()
    hashes = {"config_hash": digest_json(config), "checkpoint": checkpoint,
              "checkpoint_sha256": sha256(run / checkpoint)}
    return model, config, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "formal"))
    parser.add_argument("--fold", choices=FOLDS, default="eth")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=0)
    parser.add_argument("--runs", type=Path, help="Folder above fold/seed directories")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--phase", choices=("all", "train", "export", "runtime"), default="all")
    parser.add_argument("--condition", choices=[tag for tag, _, _ in CONDITIONS],
                        help="Export one recorded condition for replay")
    parser.add_argument("--export-to", type=Path, help="Write a replay outside the original run")
    args = parser.parse_args()
    if args.stage == "pilot" and (args.fold, args.seed) != ("eth", 0):
        parser.error("The group pilot is ETH seed 0")
    if (args.condition or args.export_to) and args.phase != "export":
        parser.error("Replay options require --phase export")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    runs = args.runs or ARTIFACTS / args.stage / "runs"
    run = runs / args.fold / f"seed_{args.seed}"
    started = time.monotonic()
    if args.phase in ("all", "train"):
        fit(run, args.stage, args.fold, args.seed, args.data, device)
    if args.phase != "train":
        model, config, hashes = restore(run, device)
        neutral = load_split(args.fold, "test", str(args.data))
        test = GraphWindows(neutral, max_neighbors=config["max_neighbors"])
        if args.phase in ("all", "export"):
            destination = args.export_to or run
            for tag, _, _ in CONDITIONS:
                if args.condition is None or tag == args.condition:
                    export_condition(model, test, destination, seed=args.seed, tag=tag,
                                     batch_size=config["export_batch_size"], device=device, run_hashes=hashes)
        if args.phase in ("all", "runtime"):
            time_model(run, model, neutral, test, args.fold, device, args.stage == "formal" and args.seed == 0)
    if args.phase == "all":
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(run / "execution.json", {
            "elapsed_seconds": time.monotonic()-started,
            "peak_rss_bytes": int(peak if sys.platform == "darwin" else 1024*peak),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        })
    print(f"OWNER_RUN_OK {args.stage} {args.fold} {args.seed} {args.phase}", flush=True)


if __name__ == "__main__":
    main()
