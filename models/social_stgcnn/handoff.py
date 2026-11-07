"""Bind the compact group contribution to the audited private review archive."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from li_junhao_trajectronpp.independent_tpp.protocol import sha256, write_json

CONTRIBUTION = ROOT / "submission/contributions/social_stgcnn"
ARTIFACT = ROOT / "artifacts/social_stgcnn"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--out", type=Path, default=CONTRIBUTION)
    parser.add_argument("--download-url", required=True, help="Private group review asset URL")
    args = parser.parse_args()
    art, out = args.artifact, args.out
    evidence = out / "evidence"
    audit = read(out / "benchmark_audit.json")
    if audit["status"] != "passed" or audit["results_sha256"] != sha256(out / "results.csv"):
        raise ValueError("The formal audit does not identify this result table")
    archive = read(art / "archive_receipt.json")
    replay = read(art / "reproduction_check.json")
    if replay["status"] != "passed" or replay["verified_bundles"] != 96:
        raise ValueError("The extracted 90 formal and six pilot bundles need a passing checkpoint replay")
    selection = read(art / "selection_check.json")
    if selection["passed"] is not True or selection["summary"]["runs_passed"] != 15:
        raise ValueError("All fifteen selected weights must reproduce their clean-validation likelihoods")
    for name in ("pilot_audit.json", "reproduction_check.json", "selection_check.json"):
        shutil.copyfile(art / name, evidence / name)
    with (evidence / "runtime.csv").open(newline="") as handle:
        timed = [r for r in csv.DictReader(handle) if r["scope"] == "run"]
    if len(timed) != 15 or len({r["host"] for r in timed}) != 1:
        raise ValueError("The review table needs fifteen timings measured on one host")
    timing = {name: statistics.median(float(r[name]) for r in timed) for name in
              ("preprocessing_ms", "prediction_sampling_ms", "end_to_end_ms")}
    transfer = {**archive, "url": args.download_url}
    write_json(out / "provenance.json", {
        "model": "social_stgcnn", "owner": "Zongmin Zhang", "protocol": "causal_v2",
        "runs": 15, "prediction_bundles": 90, "sample_count": 20,
        "parameters": 2208, "group_base": "df016def5890898b70d372fb0e8e2bc142121330",
        "numerical_revision": "2b2d57de6d5ff009f1388e1f0d9baf948e9db4ac",
        "prespecified_analysis_revision": "9cb6ac43b8c13bfcc3e8d65de891030fc35b8b23",
        "contract_sha256": audit["contract_sha256"], "results_sha256": audit["results_sha256"],
        "artifact": transfer, "runtime_ms": timing,
        "runtime_environment": {k: timed[0][k] for k in ("os", "architecture", "cpu", "threads",
                                                         "python", "numpy", "torch", "device")},
        "runtime_design": "Median of fifteen run medians on one physical CPU host. Each run uses 128 registered targets in batches of 32, three warmups and ten measurements.",
        "pilot_execution": read(art / "pilot/runs/eth/seed_0/execution.json"),
        "checkpoint_replay": {"case": replay["case"], "status": replay["status"],
                              "max_abs_forecast_difference_m": replay["comparison"]["samples"]["max_absolute_difference_m"]},
        "evidence": {str(path.relative_to(out)): sha256(path)
                     for folder in (evidence, out / "assets") for path in sorted(folder.glob("*")) if path.is_file()},
    })
    print("GROUP_HANDOFF_OK social_stgcnn")


if __name__ == "__main__":
    main()
