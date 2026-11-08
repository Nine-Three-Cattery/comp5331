"""Exercise review-archive source and receipt binding using temporary, synthetic run records."""

import csv
import json
from pathlib import Path
import runpy
import shutil
import tarfile
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from models.social_stgcnn.package_review import PROJECT, SHARED, package
from models.social_stgcnn.handoff import publish
from models.social_stgcnn.verify_review import (
    ARCHITECTURE, CASE, MODEL, NUMERICAL_FILES, PREFIX, REFERENCE, RUNS, verify_files, verify_trained_reference,
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ArchiveIdentityTests(unittest.TestCase):
    def test_gitless_extraction_and_rejection_of_a_mismatched_pilot(self):
        # This tests archive identities. It does not train, replay, or score a model.
        with TemporaryDirectory() as temp:
            scratch = Path(temp)
            artifacts = scratch / "artifacts"
            results = scratch / "results.csv"
            commit = "a" * 40
            protocol = runpy.run_path(str(PROJECT / SHARED / "independent_tpp/protocol.py"))
            sha256, digest_config = protocol["sha256"], protocol["digest_json"]
            contract = runpy.run_path(str(PROJECT / "benchmark/contract.py"))["fingerprint"]()
            sources = {name: sha256(PROJECT / MODEL / name) for name in NUMERICAL_FILES}
            manifests = json.loads((PROJECT / "benchmark/fixtures/manifests.json").read_text())["folds"]
            scored, validated = [], []
            metrics = ("min_ade", "mean_ade", "min_fde", "mean_fde")
            for stage in ("pilot", "formal"):
                fold_seeds = [("eth", 0)] if stage == "pilot" else [
                    (f, s) for f in protocol["FOLDS"] for s in protocol["SEEDS"]]
                for fold, seed in fold_seeds:
                    base = artifacts / stage / "runs" / fold / f"seed_{seed}"
                    config = {"model": "social_stgcnn", "architecture": ARCHITECTURE,
                              "reference_revision": REFERENCE, "protocol": "causal_v2",
                              "stage": stage, "fold": fold, "seed": seed, "source_sha256": sources,
                              "val_identity": {"manifest_hash": manifests[fold]["val"]["sha256"]}}
                    write(base / "config.json", config)
                    choice = {"checkpoint": "best.pt", "epoch": 1}
                    write(base / "checkpoint_selection.json", {"selected": choice, "selection_split": "val",
                                                              "test_used_for_selection": False})
                    (base / "best.pt").write_bytes(f"synthetic fixture {stage}/{fold}/{seed}".encode())
                    (base / "training.csv").write_text("epoch\n1\n")
                    for name in ("execution.json", "runtime.json"):
                        write(base / name, {})
                    checkpoint_hash = sha256(base / "best.pt")
                    if stage == "formal":
                        validated.append({"fold": fold, "seed": seed, "passed": True, "selected_epoch": 1,
                                          "checkpoint_sha256": checkpoint_hash,
                                          "validation_manifest_sha256": manifests[fold]["val"]["sha256"]})
                    for tag, kind, level in protocol["CONDITIONS"]:
                        output = base / "evaluation" / tag / f"predictions_test_{tag}.npz"
                        output.parent.mkdir(parents=True)
                        metadata = {"fold": fold, "split": "test", "seed": seed, "condition": kind,
                                    "level": level, "sample_count": 20, "config_hash": digest_config(config),
                                    "checkpoint": "best.pt", "checkpoint_sha256": checkpoint_hash,
                                    "manifest_hash": manifests[fold]["test"]["sha256"]}
                        np.savez(output, metadata=json.dumps(metadata))
                        if stage == "formal":
                            scored.append({"fold": fold, "seed": seed, "condition": tag,
                                           "prediction_sha256": sha256(output), "checkpoint_sha256": checkpoint_hash,
                                           "config_hash": digest_config(config), "manifest_hash": metadata["manifest_hash"],
                                           **{key + "_mean": 0 for key in metrics}})
            write_csv(results, scored)
            result_hash = sha256(results)
            audit = {"status": "passed", "model": "social_stgcnn", "protocol": "causal_v2", "contract_sha256": contract}
            write(artifacts / "benchmark_audit.json", {**audit, "prediction_bundles": 90, "results_sha256": result_hash})
            write(artifacts / "pilot_audit.json", {**audit, "prediction_bundles": 6})
            write(artifacts / "source_parity.json", {"passed": True, "release": REFERENCE, "model_sha256": sources["model.py"],
                                                      "owner_active_parameters": 6254, "upstream_registered_parameters": 7563,
                                                      "cases": ["synthetic fixture"]})
            fixed = json.loads((PROJECT / "benchmark/fixtures/runtime_windows.json").read_text())
            trained = {"passed": True, "upstream_revision": REFERENCE, "runtime_fixture_index_zero_based": 64,
                       "absolute_coordinate_tolerance_m": 1e-4, "samples_per_case": 20,
                       "summary": {"cases": 45, "passed": 45, "failed": 0, "max_abs_coordinate_error_m": 1e-6},
                       "cases": [{**row, "passed": True, "max_abs_coordinate_error_m": 1e-6,
                                  "window_id": fixed[row["fold"]][64]} for row in scored
                                 if row["condition"] in ("clean_0", "noise_020", "missing_2")]}
            write(artifacts / "trained_reference.json", trained)
            write(artifacts / "selection_check.json", {"passed": True, "split": "val", "runs": validated,
                                                       "summary": {"runs_passed": 15}})
            write(artifacts / "fresh_source_reproduction_check.json", {
                "scope": "fresh_source", "status": "passed", "case": CASE, "numerical_source_commit": commit,
                "architecture": ARCHITECTURE, "source_sha256": sources, "benchmark_contract_sha256": contract,
                "results_sha256": result_hash, "config_hash": scored[0]["config_hash"],
                "checkpoint_sha256": scored[0]["checkpoint_sha256"], "original_prediction_sha256": scored[0]["prediction_sha256"],
                "comparison": {k: {"exact": True, "max_absolute_difference_m": 0} for k in (
                    "samples", "ground_truth", "observed", "observed_mask", "records", "metadata", "manifest_order", "sample_budget")},
            })

            tar_path = scratch / (PREFIX + ".tar")
            receipt = package(PROJECT, artifacts, results, tar_path, commit)
            self.assertEqual(receipt["numerical_source_commit"], commit)
            with tarfile.open(tar_path) as archive:
                archive.extractall(scratch, filter="data")
            bundle = scratch / PREFIX
            inventory = verify_files(bundle, bundle)
            self.assertEqual((inventory["formal_bundles"], inventory["pilot_bundles"]), (90, 6))
            for path in (MODEL / "check_reference.py", MODEL / "check_trained_reference.py", RUNS / "trained_reference.json",
                         MODEL / "review_metrics.py", MODEL / "own_unit_tests/test_analysis.py",
                         SHARED / "scripts/analyze_independent_diagnostics.py", RUNS / "source_parity.json",
                         RUNS / "fresh_source_reproduction_check.json"):
                self.assertTrue((bundle / path).is_file(), path)

            write(artifacts / "archive_receipt.json", receipt)
            contribution = scratch / "contribution"
            evidence = contribution / "evidence"
            evidence.mkdir(parents=True)
            (contribution / "assets").mkdir()
            shutil.copyfile(results, contribution / "results.csv")
            write_csv(evidence / "review_metrics.csv", [
                {**{key: row[key] for key in ("fold", "seed", "condition")},
                 **{key: row[key + "_mean"] for key in metrics}} for row in scored])
            setup = {key: "fixture" for key in ("host", "os", "architecture", "cpu", "threads", "python", "numpy",
                                                "torch", "device", "gpu", "cuda", "driver")}
            write_csv(evidence / "runtime.csv", [
                {"fold": row["fold"], "seed": row["seed"], "scope": "run", **setup,
                 "preprocessing_ms": 1, "prediction_sampling_ms": 2, "end_to_end_ms": 3} for row in validated])
            publish(artifacts, contribution, "https://example.invalid/" + tar_path.name)
            provenance = json.loads((contribution / "provenance.json").read_text())
            self.assertEqual((provenance["parameters"], provenance["upstream_registered_parameters"]), (6254, 7563))
            self.assertEqual(provenance["checkpoint_replay"]["scope"], "fresh_source")
            self.assertEqual(provenance["numerical_revision"], commit)
            self.assertEqual(provenance["trained_reference"]["max_abs_coordinate_error_m"], 1e-6)
            self.assertEqual(json.loads((evidence / "trained_reference.json").read_text()), trained)
            trained["cases"][0]["prediction_sha256"] = "different formal bundle"
            with self.assertRaisesRegex(ValueError, "Trained reference case differs"):
                verify_trained_reference(trained, {(row["fold"], row["seed"], row["condition"]): row for row in scored}, fixed)

            pilot_path = artifacts / "pilot/runs/eth/seed_0/config.json"
            pilot = json.loads(pilot_path.read_text())
            pilot["source_sha256"]["run.py"] = "0" * 64
            write(pilot_path, pilot)
            for path in pilot_path.parent.glob("evaluation/*/*.npz"):
                with np.load(path) as original:
                    metadata = json.loads(str(original["metadata"]))
                metadata["config_hash"] = digest_config(pilot)
                np.savez(path, metadata=json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "frozen numerical source"):
                package(PROJECT, artifacts, results, tar_path, commit)
            self.assertEqual(sha256(tar_path), receipt["sha256"])


if __name__ == "__main__":
    unittest.main()
