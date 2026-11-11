"""Bind the compact public contribution to its audited full-scene review archive."""

import argparse
import csv
import json
import math
from pathlib import Path
import runpy
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.social_stgcnn.verify_review import (
    ARCHITECTURE, PREFIX, REFERENCE, read, result_rows, revision, source_identity, verify_supplemental,
)

CONTRIBUTION = ROOT / "submission/contributions/social_stgcnn"
ARTIFACT = ROOT / "artifacts/social_stgcnn"


def publish(art, out, download_url):
    protocol = runpy.run_path(str(ROOT / "li_junhao_trajectronpp/independent_tpp/protocol.py"))
    sha256 = protocol["sha256"]
    evidence = out / "evidence"
    result_path = out / "results.csv"
    results_hash = sha256(result_path)
    scored = result_rows(result_path, protocol)
    audit = read(art / "benchmark_audit.json")
    if (audit["status"], audit["model"], audit["protocol"], audit["prediction_bundles"], audit["results_sha256"]) != (
            "passed", "social_stgcnn", "causal_v2", 90, results_hash):
        raise ValueError("The formal benchmark audit does not identify this result table")
    archive = read(art / "archive_receipt.json")
    commit = revision(archive["numerical_source_commit"])
    if (archive["name"], archive["architecture"], archive["reference_revision"],
            archive["results_sha256"]) != (PREFIX + ".tar", ARCHITECTURE, REFERENCE, results_hash):
        raise ValueError("Archive receipt does not identify the new full-scene release and its scores")
    if download_url.split("?")[0].rsplit("/", 1)[-1] != archive["name"]:
        raise ValueError("The review asset URL must identify the packaged archive filename")
    sources = source_identity(read(art / "formal/runs/eth/seed_0/config.json"))
    parity, selection, replay, trained = verify_supplemental(
        art, scored, commit, sources, audit["contract_sha256"], results_hash,
        read(ROOT / "benchmark/fixtures/runtime_windows.json"))
    with (evidence / "review_metrics.csv").open(newline="") as stream:
        reviews = list(csv.DictReader(stream))
    by_cell = {(r["fold"], int(r["seed"]), r["condition"]): r for r in reviews}
    if len(reviews) != len(scored) or by_cell.keys() != scored.keys():
        raise ValueError("The owner metric table must cover the same 90 cells as the full-scene results")
    for key, owner in by_cell.items():
        for metric in ("min_ade", "mean_ade", "min_fde", "mean_fde"):
            if not math.isclose(float(owner[metric]), float(scored[key][metric + "_mean"]), rel_tol=0, abs_tol=1e-7):
                raise ValueError(f"Owner metric table refers to different forecasts at {key}/{metric}")

    with (evidence / "runtime.csv").open(newline="") as stream:
        timed = [r for r in csv.DictReader(stream) if r["scope"] == "run"]
    setup_fields = ("host", "os", "architecture", "cpu", "threads", "python", "numpy", "torch", "device", "gpu", "cuda", "driver")
    if (len(timed) != 15 or {(r["fold"], int(r["seed"])) for r in timed} != {(f,s) for f,s,_ in scored}
            or len({tuple(r[k] for k in setup_fields) for r in timed}) != 1):
        raise ValueError("The group headline runtime requires all fifteen runs measured on one matched device and host")
    timing = {name: statistics.median(float(r[name]) for r in timed)
              for name in ("preprocessing_ms", "prediction_sampling_ms", "end_to_end_ms")}

    # Use the replay that compared a fresh export of the frozen revision with
    # the original forecasts. This also replaces the earlier variant's receipt.
    for src, dst in (("pilot_audit.json", "pilot_audit.json"), ("source_parity.json", "source_parity.json"),
                     ("selection_check.json", "selection_check.json"), ("trained_reference.json", "trained_reference.json"),
                     ("fresh_source_reproduction_check.json", "reproduction_check.json")):
        shutil.copyfile(art / src, evidence / dst)
    shutil.copyfile(art / "benchmark_audit.json", out / "benchmark_audit.json")
    transfer = {key: archive[key] for key in ("name", "bytes", "sha256")}
    transfer["url"] = download_url
    protocol["write_json"](out / "provenance.json", {
        "model": "social_stgcnn", "owner": "Zongmin Zhang", "protocol": protocol["VERSION"],
        "architecture": ARCHITECTURE, "reference_revision": REFERENCE, "numerical_revision": commit,
        "parameters": parity["owner_active_parameters"],
        "upstream_registered_parameters": parity["upstream_registered_parameters"],
        "runs": 15, "prediction_bundles": 90, "pilot_prediction_bundles": 6, "sample_count": 20,
        "contract_sha256": audit["contract_sha256"], "results_sha256": results_hash,
        "artifact": transfer, "runtime_ms": timing,
        "runtime_environment": {k: timed[0][k] for k in setup_fields},
        "runtime_design": ("Median of fifteen same-host run medians. Each run measures 128 registered targets "
                           "in batches of 32, with three warmups and ten timed repetitions."),
        "pilot_execution": read(art / "pilot/runs/eth/seed_0/execution.json"),
        "source_parity": {"passed": parity["passed"], "fixture_count": len(parity["cases"]),
                          "model_sha256": parity["model_sha256"], "receipt": "evidence/source_parity.json"},
        "trained_reference": {**trained["summary"], "runtime_fixture_index_zero_based": trained["runtime_fixture_index_zero_based"],
                              "absolute_coordinate_tolerance_m": trained["absolute_coordinate_tolerance_m"],
                              "upstream_revision": trained["upstream_revision"], "receipt": "evidence/trained_reference.json"},
        "checkpoint_selection": {"runs_passed": selection["summary"]["runs_passed"], "receipt": "evidence/selection_check.json"},
        "checkpoint_replay": {"case": replay["case"], "status": replay["status"], "scope": replay["scope"],
                              "max_abs_forecast_difference_m": replay["comparison"]["samples"]["max_absolute_difference_m"],
                              "receipt": "evidence/reproduction_check.json"},
        "evidence": {str(p.relative_to(out)): sha256(p) for folder in (evidence, out / "assets")
                     for p in sorted(folder.iterdir()) if p.is_file()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--out", type=Path, default=CONTRIBUTION)
    parser.add_argument("--download-url", required=True, help="Published private review asset URL")
    args = parser.parse_args()
    publish(args.artifact, args.out, args.download_url)
    print("GROUP_HANDOFF_OK social_stgcnn")


if __name__ == "__main__":
    main()
