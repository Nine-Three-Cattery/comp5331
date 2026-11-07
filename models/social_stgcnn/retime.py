"""Time all fifteen selected checkpoints in one Linux CPU allocation."""

import argparse
import shutil

import torch

from .run import ARTIFACTS, DEFAULT_DATA, FOLDS, SEEDS, GraphWindows, load_split, restore, time_model


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ARTIFACTS / "formal/runs")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cpu")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for fold in FOLDS:
        store = load_split(fold, "test", str(args.data))
        windows = GraphWindows(store)
        for seed in SEEDS:
            run = args.runs / fold / f"seed_{seed}"
            original = run / "runtime_training_host.json"
            if not original.exists():
                shutil.copy2(run / "runtime.json", original)
            model, _, _ = restore(run, device)
            time_model(run, model, store, windows, fold, device, seed == 0)
            print(f"MATCHED_TIMING_OK {fold} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
