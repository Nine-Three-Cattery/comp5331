"""Package the audited Social-STGCNN matrix for private course review."""

import argparse
import io
import json
from pathlib import Path
import runpy
import sys
import tarfile
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from models.social_stgcnn.verify_review import (
    ARCHITECTURE, MODEL, PREFIX, RECEIPTS, REFERENCE, RESULTS, pilot_checkpoint, result_rows, revision,
    source_identity, verify_supplemental,
)

ARTIFACTS = Path("artifacts/social_stgcnn")
SHARED = Path("li_junhao_trajectronpp")


def shared_paths():
    yield Path(".gitignore")
    yield Path("Makefile")
    yield Path("CONTRIBUTING.md")
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
                 "tiny_metric_check", "evaluate", "metrics", "analyze_sample_budget",
                 "analyze_independent_diagnostics", "generate_failure_analysis"):
        yield SHARED / "scripts" / f"{name}.py"


def add_stream(archive, relative, handle, size):
    # TarInfo defaults omit local account names, ownership and timestamps.
    entry = tarfile.TarInfo(f"{PREFIX}/{relative}")
    entry.size = size
    archive.addfile(entry, handle)


def package(project, artifacts, results, output, numerical_revision):
    project = project.resolve()
    commit = revision(numerical_revision)
    protocol = runpy.run_path(str(project / SHARED / "independent_tpp/protocol.py"))
    sha256 = protocol["sha256"]
    sha_config = protocol["digest_json"]
    tags = [row[0] for row in protocol["CONDITIONS"]]
    folds, seeds = protocol["FOLDS"], protocol["SEEDS"]
    contract = runpy.run_path(str(project / "benchmark/contract.py"))["fingerprint"]()
    manifest_lock = json.loads((project / "benchmark/fixtures/manifests.json").read_text())["folds"]
    files = {path: project / path for path in shared_paths()}
    # Numerical paths below come from the frozen export. Review material
    # comes from the checkout running this packager and has its own file digests.
    review_dir = Path(__file__).resolve().parent
    # The registry's display name belongs to the current review contribution.
    # The run configs record the numerical architecture and sample budget.
    files[Path("submission/models.json")] = PROJECT / "submission/models.json"
    for name in ("README.md", "REVIEW.md", "check_reference.py", "check_trained_reference.py", "check_selection.py", "review_metrics.py",
                 "analyze.py", "retime.py", "package_review.py", "verify_review.py", "handoff.py",
                 "independent_design.md", "paper_reading_notes.md", "implementation_provenance.md",
                 "source_inventory.md", "reused_fragments.csv"):
        files[MODEL / name] = review_dir / name
    for path in sorted((review_dir / "own_unit_tests").glob("test_*.py")):
        files[MODEL / "own_unit_tests" / path.name] = path
    files[Path("REVIEW.md")] = review_dir / "REVIEW.md"
    for name in RECEIPTS:
        files[ARTIFACTS / name] = artifacts / name
    files[RESULTS] = results
    scores = result_rows(results, protocol)

    # The published scores bind formal checkpoints and bundles. A new digest
    # over an unrelated run would not establish that it produced those scores.
    published = {}
    sources = source_identity(json.loads((artifacts / "pilot/runs/eth/seed_0/config.json").read_text()))
    for name, digest in sources.items():
        files[MODEL / name] = project / MODEL / name
        published[MODEL / name] = digest
    for stage in ("pilot", "formal"):
        receipt_name = "pilot_audit.json" if stage == "pilot" else "benchmark_audit.json"
        receipt = json.loads((artifacts / receipt_name).read_text())
        total = len(tags) if stage == "pilot" else len(scores)
        if (receipt["status"], receipt["model"], receipt["protocol"], receipt["prediction_bundles"],
            receipt["contract_sha256"]) != ("passed", "social_stgcnn", protocol["VERSION"], total, contract):
            raise ValueError(f"{stage} audit does not cover this benchmark and matrix")
        if stage == "formal":
            published[RESULTS] = receipt["results_sha256"]
        runs = [("eth", 0)] if stage == "pilot" else [(fold, seed) for fold in folds for seed in seeds]
        for fold, seed in runs:
            relative = Path(stage) / "runs" / fold / f"seed_{seed}"
            run = artifacts / relative
            config = json.loads((run / "config.json").read_text())
            config_hash = sha_config(config)
            selection = json.loads((run / "checkpoint_selection.json").read_text())
            if (config["stage"], config["fold"], config["seed"]) != (stage, fold, seed) or source_identity(config) != sources:
                raise ValueError(f"Run identity or frozen numerical source differs for {relative}")
            checkpoint = selection["selected"]["checkpoint"]
            if (Path(checkpoint).name != checkpoint or selection["selection_split"] != "val"
                    or selection["test_used_for_selection"] is not False):
                raise ValueError(f"No validation-selected checkpoint for {relative}")
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
                    digest = pilot_checkpoint(run / path, config_hash=config_hash, checkpoint=checkpoint,
                                              manifest_hash=manifest_lock[fold]["test"]["sha256"],
                                              kind=kind, level=level)
                    if digest != published.setdefault(base / checkpoint, digest):
                        raise ValueError(f"Pilot identity differs at {relative}/{tag}")

    runtime_windows = json.loads((project / "benchmark/fixtures/runtime_windows.json").read_text())
    parity, _, _, _ = verify_supplemental(artifacts, scores, commit, sources, contract, published[RESULTS], runtime_windows)
    digests = {name.as_posix(): sha256(path) for name, path in files.items()}
    for name, expected_hash in published.items():
        if digests[name.as_posix()] != expected_hash:
            raise ValueError(f"Payload differs from its recorded identity: {name}")
    inventory = {
        "archive_root": PREFIX, "protocol": protocol["VERSION"],
        "architecture": ARCHITECTURE, "reference_revision": REFERENCE,
        "owner_active_parameters": parity["owner_active_parameters"],
        "upstream_registered_parameters": parity["upstream_registered_parameters"],
        "numerical_source_commit": commit, "source_sha256": sources, "benchmark_contract_sha256": contract,
        "raw_release_commit": runpy.run_path(str(project / SHARED / "scripts/fetch_causal_raw.py"))["COMMIT"],
        "raw_sha256_file": (SHARED / "configs/causal_v2_raw_sha256.json").as_posix(),
        "ordered_manifest_lock": "benchmark/fixtures/manifests.json",
        "formal_runs": len(folds) * len(seeds), "formal_bundles": len(scores),
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
    return {"name": output.name, "bytes": output.stat().st_size, "sha256": sha256(output),
            "numerical_source_commit": commit, "architecture": ARCHITECTURE,
            "reference_revision": REFERENCE, "results_sha256": published[RESULTS]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--numerical-revision", required=True, help="Full commit of the frozen --project export")
    parser.add_argument("--artifacts", type=Path, help="Folder containing pilot, formal and both group receipts")
    parser.add_argument("--results", type=Path, help="Formal 90-row group result CSV")
    parser.add_argument("--output", type=Path, help="Uncompressed .tar path")
    parser.add_argument("--receipt", type=Path, help="Defaults to <artifacts>/archive_receipt.json")
    args = parser.parse_args()
    artifacts = args.artifacts or args.project / ARTIFACTS
    output = args.output or artifacts / f"{PREFIX}.tar"
    if output.name != PREFIX + ".tar":
        parser.error(f"The full-scene asset filename is {PREFIX}.tar")
    receipt = package(args.project, artifacts, args.results or args.project / RESULTS, output, args.numerical_revision)
    destination = args.receipt or artifacts / "archive_receipt.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    message = json.dumps(receipt, sort_keys=True, indent=2)
    destination.write_text(message + "\n")
    print(message)


if __name__ == "__main__":
    main()
