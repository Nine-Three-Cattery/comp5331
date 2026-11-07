"""Check the extracted review files and replay ETH seed 0 clean on Linux CPU."""

import argparse
import csv
import json
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path
import platform
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
MODEL = Path("models/social_stgcnn")
RUNS = Path("artifacts/social_stgcnn")
RESULTS = Path("submission/contributions/social_stgcnn/results.csv")


def verify_files(bundle, source):
    source = source.resolve()
    inventory = json.loads((bundle / "inventory.json").read_text())
    files = inventory["files_sha256"]
    protocol = runpy.run_path(str(source / "li_junhao_trajectronpp/independent_tpp/protocol.py"))
    sha256 = protocol["sha256"]
    for name, digest in files.items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"Changed review file: {name}")
    contract = runpy.run_path(str(source / "benchmark/contract.py"))["fingerprint"]()
    if inventory["benchmark_contract_sha256"] != contract:
        raise ValueError("Replay checkout and archived benchmark contract differ")
    tags = [tag for tag, _, _ in protocol["CONDITIONS"]]
    formal = [(fold, seed) for fold in protocol["FOLDS"] for seed in protocol["SEEDS"]]
    with (bundle / RESULTS).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored = {(r["fold"], int(r["seed"]), r["condition"]): r for r in rows}
    if len(rows) != 90 or len(scored) != 90:
        raise ValueError("Expected 90 unique formal result rows")
    bundles = set()
    for stage, runs in (("pilot", [("eth", 0)]), ("formal", formal)):
        receipt = "pilot_audit.json" if stage == "pilot" else "benchmark_audit.json"
        audit = json.loads((bundle / RUNS / receipt).read_text())
        if (audit["status"], audit["contract_sha256"], audit["prediction_bundles"]) != ("passed", contract, len(runs)*len(tags)):
            raise ValueError(f"Invalid {stage} benchmark receipt")
        if stage == "formal" and audit["results_sha256"] != files[RESULTS.as_posix()]:
            raise ValueError("The formal benchmark receipt does not identify the archived results")
        for fold, seed in runs:
            base = RUNS / stage / "runs" / fold / f"seed_{seed}"
            config = json.loads((bundle / base / "config.json").read_text())
            selection = json.loads((bundle / base / "checkpoint_selection.json").read_text())
            checkpoint = selection["selected"]["checkpoint"]
            for name in ("config.json", "checkpoint_selection.json", checkpoint, "training.csv", "runtime.json", "execution.json"):
                if (base / name).as_posix() not in files:
                    raise ValueError(f"Missing run evidence: {base / name}")
            for name, digest in config["source_sha256"].items():
                recorded = (inventory["pilot_run_py_sha256"] if stage == "pilot" and name == "run.py"
                            else files[(MODEL / name).as_posix()])
                if recorded != digest:
                    raise ValueError(f"Archived numerical sources differ from {base / 'config.json'}")
            for tag in tags:
                name = (base / "evaluation" / tag / f"predictions_test_{tag}.npz").as_posix()
                bundles.add(name)
                if stage == "formal":
                    r = scored[fold, seed, tag]
                    if (files[name], files[(base / checkpoint).as_posix()], protocol["digest_json"](config)) != (
                            r["prediction_sha256"], r["checkpoint_sha256"], r["config_hash"]):
                        raise ValueError(f"Scored run identities differ at {name}")
    if bundles != {name for name in files if name.endswith(".npz")}:
        raise ValueError("Review must contain exactly 90 formal and six pilot prediction bundles")
    clean_run = RUNS / "formal/runs/eth/seed_0"
    config = json.loads((bundle / clean_run / "config.json").read_text())
    for name, digest in config["source_sha256"].items():
        if sha256(source / MODEL / name) != digest:
            raise ValueError(f"Fresh checkout has a different numerical source: {name}")
    return inventory, config, len(bundles)


def compare_bundles(original, replay, window_ids):
    """Compare unpacked values and retain numerical differences for the receipt."""
    result = {}
    with np.load(original, allow_pickle=False) as ref, np.load(replay, allow_pickle=False) as got:
        fields = {"samples", "ground_truth", "observed", "observed_mask", "records", "metadata"}
        if set(ref.files) != fields or set(got.files) != fields:
            raise ValueError("Prediction bundles have different or incomplete fields")
        for key in ("samples", "ground_truth", "observed", "observed_mask"):
            left, right = ref[key], got[key]
            info = {"shape": list(right.shape), "dtype": str(right.dtype),
                    "exact": left.dtype == right.dtype and np.array_equal(left, right)}
            if key == "samples":
                result["sample_budget"] = {"exact": left.shape == (len(window_ids), 20, 12, 2)}
            if left.dtype != right.dtype:
                info["original_dtype"] = str(left.dtype)
            if left.shape == right.shape:
                info["different_values"] = int(np.count_nonzero(left != right))
                if key != "observed_mask":
                    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
                    info["max_absolute_difference_m"] = float(delta.max()) if np.isfinite(delta).all() else None
            else:
                info["original_shape"] = list(left.shape)
            result[key] = info
        left_records = [json.loads(r) for r in ref["records"]]
        right_records = [json.loads(r) for r in got["records"]]
        result["records"] = {"exact": left_records == right_records, "rows": len(right_records),
                             "different_rows": sum(a != b for a, b in zip_longest(left_records, right_records))}
        original_ids = [r["window_id"] for r in left_records]
        replay_ids = [r["window_id"] for r in right_records]
        result["manifest_order"] = {"exact": window_ids == original_ids == replay_ids}
        left_meta, right_meta = json.loads(str(ref["metadata"])), json.loads(str(got["metadata"]))
        result["metadata"] = {"exact": left_meta == right_meta,
                              "different_keys": sorted(k for k in left_meta.keys() | right_meta.keys()
                                                       if k not in left_meta or k not in right_meta or left_meta[k] != right_meta[k])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=PROJECT, help="Extracted review root containing inventory.json")
    parser.add_argument("--source", type=Path, help="Optional independent export of the pinned project source")
    parser.add_argument("--data", type=Path, help="Rebuilt causal_v2 data, defaults to <source>/data/causal_v2")
    parser.add_argument("--receipt", type=Path, help="Write the replay comparison as JSON")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    source = (args.source or bundle).resolve()
    data = (args.data or source / "data/causal_v2").resolve()
    env = {"os": platform.system(), "architecture": platform.machine(), "python": platform.python_version(),
           "torch": version("torch"), "numpy": version("numpy"), "pandas": version("pandas")}
    if ((env["os"], env["architecture"], sys.version_info[:2]) != ("Linux", "x86_64", (3, 12))
            or env["torch"] != "2.14.0+cpu" or env["numpy"] != "2.5.3" or env["pandas"] != "3.0.6"):
        raise SystemExit(f"Use the pinned Linux x86-64 CPU environment in REVIEW.md. Detected {env}")
    inventory, config, count = verify_files(bundle, source)
    env["cpu_threads"] = config["environment"]["threads"]
    with (data / "eth/test_manifest.csv").open(newline="") as handle:
        window_ids = [row["window_id"] for row in csv.DictReader(handle)]
    relative = Path("evaluation/clean_0/predictions_test_clean_0.npz")
    stored_runs = bundle / RUNS / "formal/runs"
    with TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(source / MODEL / "run.py"), "formal", "--phase", "export",
                        "--fold", "eth", "--seed", "0", "--condition", "clean_0", "--device", "cpu",
                        "--threads", str(env["cpu_threads"]), "--runs", str(stored_runs), "--data", str(data),
                        "--export-to", tmp], cwd=source, check=True)
        detail = compare_bundles(stored_runs / "eth/seed_0" / relative, Path(tmp) / relative, window_ids)
    passed = all(info["exact"] for info in detail.values())
    receipt = {"status": "passed" if passed else "different", "case": "formal/eth/seed_0/clean_0",
               "verified_files": len(inventory["files_sha256"]), "verified_bundles": count,
               "numerical_source_commit": inventory["numerical_source_commit"],
               "environment": env, "comparison": detail}
    message = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(message + "\n")
    print(message)
    if not passed:
        raise SystemExit("Selected-checkpoint replay differs from the archived output. See the comparison above.")


if __name__ == "__main__":
    main()
