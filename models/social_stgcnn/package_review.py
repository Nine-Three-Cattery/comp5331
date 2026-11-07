"""Package the audited Social-STGCNN matrix for private course review."""

import argparse
import csv
import io
import json
from pathlib import Path
import runpy
import tarfile
from tempfile import TemporaryDirectory

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
MODEL = Path("models/social_stgcnn")
ARTIFACTS = Path("artifacts/social_stgcnn")
RESULTS = Path("submission/contributions/social_stgcnn/results.csv")
SHARED = Path("li_junhao_trajectronpp")
PREFIX = "social_stgcnn_causal_v2_review"
NUMERICAL_COMMIT = "2b2d57de6d5ff009f1388e1f0d9baf948e9db4ac"
PILOT_COMMIT = "c37d1211104d9c125775bd4c7b5044083c04c1f1"
PILOT_RUN_SHA256 = "cf964fd94aad0ab289775c326b7e5515f84c1face976e1f3fd1b9958fef4a711"
GROUP_BASE = "df016def5890898b70d372fb0e8e2bc142121330"


def source_paths():
    yield Path(".gitignore")
    yield Path("Makefile")
    yield Path("submission/models.json")
    for name in ("README.md", "contract.py", "setup.py", "validate.py", "requirements.txt",
                 "fixtures/manifests.json", "fixtures/corruption.json", "fixtures/runtime_windows.json"):
        yield Path("benchmark") / name
    yield SHARED / "configs/causal_v2_raw_sha256.json"
    yield SHARED / "implementation_provenance.md"
    yield SHARED / "source_inventory.md"
    # The shared package initializer imports its group-owned model module.
    for name in ("__init__.py", "data.py", "model.py", "protocol.py"):
        yield SHARED / "independent_tpp" / name
    for name in ("fetch_causal_raw", "prepare_neutral_data", "compute_independent_thresholds",
                 "tiny_metric_check", "evaluate", "metrics", "analyze_sample_budget"):
        yield SHARED / "scripts" / f"{name}.py"
    for name in ("model.py", "windows.py", "export.py", "run.py", "timing.py", "configs/experiment.json",
                 "implementation_provenance.md", "source_inventory.md",
                 "reused_fragments.csv", "paper_reading_notes.md", "independent_design.md",
                 "own_unit_tests/test_model.py", "own_unit_tests/test_windows.py", "own_unit_tests/test_timing.py"):
        yield MODEL / name


def add_stream(archive, relative, handle, size):
    # TarInfo defaults omit local account names, ownership and timestamps.
    entry = tarfile.TarInfo(f"{PREFIX}/{relative}")
    entry.size = size
    archive.addfile(entry, handle)


def package(project, artifacts, results, output):
    project = project.resolve()
    protocol = runpy.run_path(str(project / SHARED / "independent_tpp/protocol.py"))
    sha256 = protocol["sha256"]
    sha_config = protocol["digest_json"]
    tags = [row[0] for row in protocol["CONDITIONS"]]
    folds, seeds = protocol["FOLDS"], protocol["SEEDS"]
    contract = runpy.run_path(str(project / "benchmark/contract.py"))["fingerprint"]()
    manifest_lock = json.loads((project / "benchmark/fixtures/manifests.json").read_text())["folds"]
    files = {path: project / path for path in source_paths()}
    # Review tooling postdates the frozen numerical commit. A clean export of
    # that commit can therefore be supplied independently of these two files.
    files[Path("REVIEW.md")] = Path(__file__).with_name("REVIEW.md")
    files[MODEL / "verify_review.py"] = Path(__file__).with_name("verify_review.py")
    files[MODEL / "retime.py"] = Path(__file__).with_name("retime.py")
    files[MODEL / "check_selection.py"] = Path(__file__).with_name("check_selection.py")
    files[RESULTS] = results
    with results.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = {(row["fold"], int(row["seed"]), row["condition"]): row for row in rows}
    expected = {(fold, seed, tag) for fold in folds for seed in seeds for tag in tags}
    if len(rows) != len(expected) or scores.keys() != expected:
        raise ValueError("The scored table must contain all 90 unique formal cells")

    # The published scores bind formal checkpoints and bundles. A new digest
    # over an unrelated run would not establish that it produced those scores.
    published = {}
    for stage in ("pilot", "formal"):
        receipt_name = "pilot_audit.json" if stage == "pilot" else "benchmark_audit.json"
        receipt = json.loads((artifacts / receipt_name).read_text())
        total = len(tags) if stage == "pilot" else len(expected)
        if (receipt["status"], receipt["model"], receipt["protocol"], receipt["prediction_bundles"],
            receipt["contract_sha256"]) != ("passed", "social_stgcnn", protocol["VERSION"], total, contract):
            raise ValueError(f"{stage} audit does not cover this benchmark and matrix")
        if stage == "formal":
            published[RESULTS] = receipt["results_sha256"]
        files[ARTIFACTS / receipt_name] = artifacts / receipt_name
        runs = [("eth", 0)] if stage == "pilot" else [(fold, seed) for fold in folds for seed in seeds]
        for fold, seed in runs:
            relative = Path(stage) / "runs" / fold / f"seed_{seed}"
            run = artifacts / relative
            config = json.loads((run / "config.json").read_text())
            config_hash = sha_config(config)
            selection = json.loads((run / "checkpoint_selection.json").read_text())
            if (config["stage"], config["fold"], config["seed"]) != (stage, fold, seed):
                raise ValueError(f"Run identity differs for {relative}")
            checkpoint = selection["selected"]["checkpoint"]
            if (Path(checkpoint).name != checkpoint or selection["selection_split"] != "val"
                    or selection["test_used_for_selection"] is not False):
                raise ValueError(f"No validation-selected checkpoint for {relative}")
            for name, digest in config["source_sha256"].items():
                if stage == "pilot" and name == "run.py":
                    if digest != PILOT_RUN_SHA256:
                        raise ValueError("Pilot entry point differs from its training and prediction revision")
                    continue
                source_path = MODEL / name
                if source_path in published and published[source_path] != digest:
                    raise ValueError(f"Run sources differ at {relative}/{name}")
                published[source_path] = digest
            base = ARTIFACTS / relative
            for name in ("config.json", "checkpoint_selection.json", checkpoint, "training.csv", "runtime.json", "execution.json"):
                files[base / name] = run / name
            for tag, kind, level in protocol["CONDITIONS"]:
                path = Path("evaluation") / tag / f"predictions_test_{tag}.npz"
                files[base / path] = run / path
                if stage == "formal":
                    row = scores[fold, seed, tag]
                    if (row["config_hash"] != config_hash or
                            row["checkpoint_sha256"] != scores[fold, seed, tags[0]]["checkpoint_sha256"]):
                        raise ValueError(f"Selected run does not match scores for {relative}/{tag}")
                    published[base / checkpoint] = row["checkpoint_sha256"]
                    published[base / path] = row["prediction_sha256"]
                else:
                    # The pilot receipt has no score CSV. Its six bundles
                    # identify the validation-selected checkpoint directly.
                    with np.load(run / path, allow_pickle=False) as predictions:
                        meta = json.loads(str(predictions["metadata"]))
                    known = {"fold": fold, "split": "test", "seed": seed, "condition": kind, "level": level,
                             "sample_count": 20, "config_hash": config_hash, "checkpoint": checkpoint,
                             "manifest_hash": manifest_lock[fold]["test"]["sha256"]}
                    if (any(meta[k] != v for k, v in known.items()) or meta["checkpoint_sha256"] !=
                            published.setdefault(base / checkpoint, meta["checkpoint_sha256"])):
                        raise ValueError(f"Pilot identity differs at {relative}/{tag}")

    digests = {name.as_posix(): sha256(path) for name, path in files.items()}
    for name, expected_hash in published.items():
        if digests[name.as_posix()] != expected_hash:
            raise ValueError(f"Payload differs from its recorded identity: {name}")
    inventory = {
        "protocol": protocol["VERSION"], "group_base_commit": GROUP_BASE,
        "numerical_source_commit": NUMERICAL_COMMIT, "benchmark_contract_sha256": contract,
        "pilot_training_and_prediction_commit": PILOT_COMMIT, "pilot_run_py_sha256": PILOT_RUN_SHA256,
        "pilot_timing_commit": NUMERICAL_COMMIT,
        "matched_timing_driver_commit": "1987673",
        "raw_release_commit": runpy.run_path(str(project / SHARED / "scripts/fetch_causal_raw.py"))["COMMIT"],
        "raw_sha256_file": (SHARED / "configs/causal_v2_raw_sha256.json").as_posix(),
        "ordered_manifest_lock": "benchmark/fixtures/manifests.json",
        "formal_runs": len(folds) * len(seeds), "formal_bundles": len(expected),
        "pilot_runs": 1, "pilot_bundles": len(tags), "files_sha256": digests,
    }
    manifest = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        pending = Path(tmp) / "review.tar"
        with tarfile.open(pending, "w") as archive:
            for name, path in sorted(files.items()):
                with path.open("rb") as handle:
                    add_stream(archive, name.as_posix(), handle, path.stat().st_size)
            add_stream(archive, "inventory.json", io.BytesIO(manifest), len(manifest))
        pending.replace(output)
    return {"name": output.name, "bytes": output.stat().st_size, "sha256": sha256(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--artifacts", type=Path, help="Folder containing pilot, formal and both group receipts")
    parser.add_argument("--results", type=Path, help="Formal 90-row group result CSV")
    parser.add_argument("--output", type=Path, help="Uncompressed .tar path")
    args = parser.parse_args()
    artifacts = args.artifacts or args.project / ARTIFACTS
    output = args.output or artifacts / f"{PREFIX}.tar"
    if output.suffix != ".tar":
        parser.error("The review archive uses an uncompressed .tar around the compressed prediction files")
    print(json.dumps(package(args.project, artifacts, args.results or args.project / RESULTS, output), sort_keys=True))


if __name__ == "__main__":
    main()
