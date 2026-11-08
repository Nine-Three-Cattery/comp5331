"""Verify the full-scene review artifact and reproduce its selected ETH clean forecasts."""

import argparse
import csv
import json
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path
import platform
import re
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
MODEL = Path("models/social_stgcnn")
RUNS = Path("artifacts/social_stgcnn")
RESULTS = Path("submission/contributions/social_stgcnn/results.csv")
PREFIX = "social_stgcnn_full_scene_causal_v2_review"
ARCHITECTURE = "released_full_scene"
REFERENCE = "333d3a57b4d2705e129b21aefefa09c79b2b9ae1"
NUMERICAL_FILES = {"model.py", "windows.py", "export.py", "run.py", "timing.py", "configs/experiment.json"}
RECEIPTS = ("pilot_audit.json", "benchmark_audit.json", "selection_check.json",
            "source_parity.json", "fresh_source_reproduction_check.json")
CASE = "formal/eth/seed_0/clean_0"
CASE_BUNDLE = Path("evaluation/clean_0/predictions_test_clean_0.npz")


def read(path):
    return json.loads(path.read_text())


def revision(value):
    if not value or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Supply the full 40-character commit for the frozen numerical source")
    return value


def source_identity(config):
    if (config["model"], config["architecture"], config["reference_revision"], config["protocol"]) != (
            "social_stgcnn", ARCHITECTURE, REFERENCE, "causal_v2"):
        raise ValueError("Run configuration does not identify the full-scene release")
    sources = config["source_sha256"]
    if not NUMERICAL_FILES <= sources.keys():
        raise ValueError("Run configuration is missing frozen numerical source files")
    return sources


def result_rows(path, protocol):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_cell = {(r["fold"], int(r["seed"]), r["condition"]): r for r in rows}
    cells = {(f, s, tag) for f in protocol["FOLDS"] for s in protocol["SEEDS"]
             for tag, _, _ in protocol["CONDITIONS"]}
    if len(rows) != len(cells) or by_cell.keys() != cells:
        raise ValueError("The formal score table must contain the 90 unique registered cells")
    return by_cell


def pilot_checkpoint(path, *, config_hash, checkpoint, manifest_hash, kind, level):
    with np.load(path, allow_pickle=False) as bundle:
        meta = json.loads(str(bundle["metadata"]))
    expected = {"fold": "eth", "split": "test", "seed": 0, "condition": kind, "level": level, "sample_count": 20,
                "config_hash": config_hash, "checkpoint": checkpoint, "manifest_hash": manifest_hash}
    if any(meta[k] != v for k, v in expected.items()):
        raise ValueError(f"Pilot prediction identity differs at {path}")
    return meta["checkpoint_sha256"]


def verify_supplemental(artifacts, scored, commit, sources, contract, result_hash):
    parity = read(artifacts / "source_parity.json")
    if (parity["passed"], parity["release"], parity["model_sha256"], parity["owner_active_parameters"],
            parity["upstream_registered_parameters"]) != (True, REFERENCE, sources["model.py"], 6254, 7563):
        raise ValueError("Source parity receipt does not identify this full-scene numerical model")

    selection = read(artifacts / "selection_check.json")
    selected = {(row["fold"], row["seed"]): row for row in selection["runs"]}
    expected = {(fold, seed) for fold, seed, _ in scored}
    if (selection["passed"] is not True or selection["split"] != "val"
            or selection["summary"]["runs_passed"] != len(expected)
            or len(selection["runs"]) != len(expected) or selected.keys() != expected):
        raise ValueError("Validation replay must cover all fifteen formal runs")
    for (fold, seed), row in selected.items():
        run = artifacts / "formal/runs" / fold / f"seed_{seed}"
        config = read(run / "config.json")
        choice = read(run / "checkpoint_selection.json")["selected"]
        if (row["passed"] is not True or row["selected_epoch"] != choice["epoch"]
                or row["checkpoint_sha256"] != scored[fold, seed, "clean_0"]["checkpoint_sha256"]
                or row["validation_manifest_sha256"] != config["val_identity"]["manifest_hash"]):
            raise ValueError(f"Selected checkpoint or validation manifest differs for {fold}/{seed}")
    replay = read(artifacts / "fresh_source_reproduction_check.json")
    record = scored["eth", 0, "clean_0"]
    expected_fields = {"scope": "fresh_source", "status": "passed", "case": CASE,
                       "numerical_source_commit": commit, "architecture": ARCHITECTURE, "source_sha256": sources,
                       "benchmark_contract_sha256": contract, "results_sha256": result_hash,
                       "config_hash": record["config_hash"], "checkpoint_sha256": record["checkpoint_sha256"],
                       "original_prediction_sha256": record["prediction_sha256"]}
    if any(replay.get(k) != v for k, v in expected_fields.items()):
        raise ValueError("The fresh-source ETH replay does not identify this numerical revision and scored run")
    comparisons = ("samples", "ground_truth", "observed", "observed_mask", "records", "metadata",
                   "manifest_order", "sample_budget")
    if any(replay["comparison"][key]["exact"] is not True for key in comparisons):
        raise ValueError("The fresh-source ETH replay differs from the archived predictions")
    return parity, selection, replay


def verify_files(bundle, source):
    source = source.resolve()
    inventory = read(bundle / "inventory.json")
    commit = revision(inventory["numerical_source_commit"])
    protocol = runpy.run_path(str(source / "li_junhao_trajectronpp/independent_tpp/protocol.py"))
    sha256 = protocol["sha256"]
    files = inventory["files_sha256"]
    for name, digest in files.items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"Changed review file: {name}")
    contract = runpy.run_path(str(source / "benchmark/contract.py"))["fingerprint"]()
    if (inventory["archive_root"], inventory["architecture"], inventory["reference_revision"],
            inventory["owner_active_parameters"], inventory["upstream_registered_parameters"],
            inventory["benchmark_contract_sha256"]) != (PREFIX, ARCHITECTURE, REFERENCE, 6254, 7563, contract):
        raise ValueError("Review inventory has the wrong model or group benchmark identity")
    scored = result_rows(bundle / RESULTS, protocol)
    artifacts = bundle / RUNS
    sources = inventory["source_sha256"]
    if any(files[(MODEL / name).as_posix()] != digest for name, digest in sources.items()):
        raise ValueError("Inventoried numerical sources differ from those used to train the checkpoints")
    for name in RECEIPTS:
        if (RUNS / name).as_posix() not in files:
            raise ValueError(f"Receipt missing from the archive inventory: {name}")
    tags = [tag for tag, _, _ in protocol["CONDITIONS"]]
    formal = [(fold, seed) for fold in protocol["FOLDS"] for seed in protocol["SEEDS"]]
    if (inventory["formal_runs"], inventory["formal_bundles"], inventory["pilot_runs"], inventory["pilot_bundles"]) != (
            len(formal), len(scored), 1, len(tags)):
        raise ValueError("Archive inventory counts differ from the registered matrix")
    manifests = read(bundle / inventory["ordered_manifest_lock"])["folds"]
    bundles = set()
    for stage, runs in (("pilot", [("eth", 0)]), ("formal", formal)):
        audit = read(artifacts / ("pilot_audit.json" if stage == "pilot" else "benchmark_audit.json"))
        if (audit["status"], audit["model"], audit["protocol"], audit["contract_sha256"], audit["prediction_bundles"]) != (
                "passed", "social_stgcnn", "causal_v2", contract, len(runs) * len(tags)):
            raise ValueError(f"Invalid {stage} group benchmark receipt")
        if stage == "formal" and audit["results_sha256"] != files[RESULTS.as_posix()]:
            raise ValueError("The formal audit does not identify the archived results")
        for fold, seed in runs:
            base = RUNS / stage / "runs" / fold / f"seed_{seed}"
            config = read(bundle / base / "config.json")
            if source_identity(config) != sources or (config["stage"], config["fold"], config["seed"]) != (stage, fold, seed):
                raise ValueError(f"Numerical source or run identity mismatch at {base}")
            selection = read(bundle / base / "checkpoint_selection.json")
            if selection["selection_split"] != "val" or selection["test_used_for_selection"] is not False:
                raise ValueError(f"Invalid checkpoint selection at {base}")
            checkpoint = selection["selected"]["checkpoint"]
            for name in ("config.json", "checkpoint_selection.json", checkpoint, "training.csv", "runtime.json", "execution.json"):
                if (base / name).as_posix() not in files:
                    raise ValueError(f"Missing run evidence: {base / name}")
            config_hash = protocol["digest_json"](config)
            for tag, kind, level in protocol["CONDITIONS"]:
                name = (base / "evaluation" / tag / f"predictions_test_{tag}.npz").as_posix()
                bundles.add(name)
                if stage == "formal":
                    row = scored[fold, seed, tag]
                    if (files[name], files[(base / checkpoint).as_posix()], config_hash) != (
                            row["prediction_sha256"], row["checkpoint_sha256"], row["config_hash"]):
                        raise ValueError(f"Scored identity differs for {name}")
                elif pilot_checkpoint(bundle / name, config_hash=config_hash, checkpoint=checkpoint,
                                      manifest_hash=manifests[fold]["test"]["sha256"], kind=kind,
                                      level=level) != files[(base / checkpoint).as_posix()]:
                    raise ValueError(f"Pilot selected checkpoint differs for {name}")
    if bundles != {name for name in files if name.endswith(".npz")}:
        raise ValueError("Review must contain exactly 90 formal and six pilot forecasts")
    verify_supplemental(artifacts, scored, commit, sources, contract, files[RESULTS.as_posix()])
    return inventory


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


def replay(source, artifacts, data, results, commit, scope):
    protocol = runpy.run_path(str(source / "li_junhao_trajectronpp/independent_tpp/protocol.py"))
    sha256 = protocol["sha256"]
    contract = runpy.run_path(str(source / "benchmark/contract.py"))["fingerprint"]()
    audit = read(artifacts / "benchmark_audit.json")
    result_hash = sha256(results)
    if audit["status"] != "passed" or (audit["contract_sha256"], audit["results_sha256"]) != (contract, result_hash):
        raise ValueError("Source checkout, score table and formal group receipt disagree")
    record = result_rows(results, protocol)["eth", 0, "clean_0"]
    stored_runs = artifacts / "formal/runs"
    run = stored_runs / "eth/seed_0"
    config = read(run / "config.json")
    sources = source_identity(config)
    for name, digest in sources.items():
        if sha256(source / MODEL / name) != digest:
            raise ValueError(f"Fresh numerical source differs from the scored run: {name}")
    selection = read(run / "checkpoint_selection.json")
    checkpoint = selection["selected"]["checkpoint"]
    if (protocol["digest_json"](config), sha256(run / checkpoint), sha256(run / CASE_BUNDLE)) != (
            record["config_hash"], record["checkpoint_sha256"], record["prediction_sha256"]):
        raise ValueError("Selected ETH checkpoint or original bundle differs from the formal score row")
    env = {"os": platform.system(), "architecture": platform.machine(), "python": platform.python_version(),
           "torch": version("torch"), "numpy": version("numpy"), "pandas": version("pandas"),
           "cpu_threads": config["environment"]["threads"]}
    if ((env["os"], env["architecture"], sys.version_info[:2]) != ("Linux", "x86_64", (3, 12))
            or env["torch"] != "2.14.0+cpu" or env["numpy"] != "2.5.3" or env["pandas"] != "3.0.6"):
        raise SystemExit(f"Use the reference CPU environment in REVIEW.md. Detected {env}")
    with (data / "eth/test_manifest.csv").open(newline="") as stream:
        window_ids = [row["window_id"] for row in csv.DictReader(stream)]
    with TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(source / MODEL / "run.py"), "formal", "--phase", "export",
                        "--fold", "eth", "--seed", "0", "--condition", "clean_0", "--device", "cpu",
                        "--threads", str(env["cpu_threads"]), "--runs", str(stored_runs), "--data", str(data),
                        "--export-to", tmp], cwd=source, check=True)
        detail = compare_bundles(run / CASE_BUNDLE, Path(tmp) / CASE_BUNDLE, window_ids)
    return {"status": "passed" if all(info["exact"] for info in detail.values()) else "different",
            "scope": scope, "case": CASE, "numerical_source_commit": commit, "architecture": ARCHITECTURE,
            "source_sha256": sources, "benchmark_contract_sha256": contract, "results_sha256": result_hash,
            "config_hash": record["config_hash"], "checkpoint_sha256": record["checkpoint_sha256"],
            "original_prediction_sha256": record["prediction_sha256"], "environment": env, "comparison": detail}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--artifacts", type=Path, help="Pre-package replay using a fresh numerical source export")
    mode.add_argument("--bundle", type=Path, help="Extracted review root, the default verification mode")
    parser.add_argument("--source", type=Path, help="Fresh source export, or the extracted review source by default")
    parser.add_argument("--data", type=Path, help="Rebuilt causal_v2 data directory")
    parser.add_argument("--results", type=Path, help="Official 90-row score table for pre-package replay")
    parser.add_argument("--numerical-revision", help="Full commit of the externally exported numerical source")
    parser.add_argument("--receipt", type=Path, help="Write the measured comparison to this path")
    args = parser.parse_args()
    if args.artifacts:
        if not args.source or not args.numerical_revision:
            parser.error("Pre-package replay needs --source and --numerical-revision")
        artifacts = args.artifacts.resolve()
        source = args.source.resolve()
        results = (args.results or PROJECT / RESULTS).resolve()
        commit = revision(args.numerical_revision)
        inventory = None
    else:
        if args.results or args.numerical_revision:
            parser.error("Extracted verification obtains its score table and revision from the bundle")
        bundle = (args.bundle or PROJECT).resolve()
        source = (args.source or bundle).resolve()
        inventory = verify_files(bundle, source)
        commit = inventory["numerical_source_commit"]
        artifacts, results = bundle / RUNS, bundle / RESULTS
    data = (args.data or source / "data/causal_v2").resolve()
    receipt = replay(source, artifacts, data, results, commit, "extracted_archive" if inventory else "fresh_source")
    receipt["verified_bundles"] = inventory["formal_bundles"] + inventory["pilot_bundles"] if inventory else 1
    if inventory:
        receipt["verified_files"] = len(inventory["files_sha256"])
    message = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
    destination = args.receipt or (artifacts / "fresh_source_reproduction_check.json" if inventory is None else None)
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(message + "\n")
    print(message)
    if receipt["status"] != "passed":
        raise SystemExit("Checkpoint replay differs from its original bundle. The receipt contains the differences.")


if __name__ == "__main__":
    main()
